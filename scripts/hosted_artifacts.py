"""Bind, verify and promote an exact hosted bundle. No build or upload operation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import tomllib

REPOSITORY = 'bledden/triton-msl'
WORKFLOW = '.github/workflows/qualify-release.yml'
HEX = re.compile(r'[0-9a-f]{64}\Z')
SHA = re.compile(r'[0-9a-f]{40}\Z')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_files(version):
    require(isinstance(version, str) and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+)?', version), 'version')
    stem = 'triton_msl-' + version
    return {stem + '-cp314-cp314-macosx_15_0_arm64.whl', stem + '-py3-none-any.whl', stem + '.tar.gz'}


def files(dist, version):
    dist = Path(dist)
    require({p.name for p in dist.iterdir()} == expected_files(version), 'bundle file population')
    result = {}
    for p in sorted(dist.iterdir()):
        require(p.is_file() and not p.is_symlink(), 'bundle must contain regular files')
        result[p.name] = {'sha256': digest(p), 'size': p.stat().st_size}
    return result


def source_digest(source):
    path = Path(source) / 'scripts/check_engineering.py'
    spec = importlib.util.spec_from_file_location('engineering_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_digest(Path(source))


def validate_record(record, tag):
    require(type(record) is dict and set(record) == {
        'schema', 'repository', 'run_id', 'artifact_id', 'artifact_name', 'source_sha',
        'source_digest', 'version', 'files', 'qualification'}, 'record keys')
    require(type(record['schema']) is int and record['schema'] == 1, 'schema')
    require(record['repository'] == REPOSITORY, 'repository')
    for key in ('run_id', 'artifact_id'):
        require(type(record[key]) is int and record[key] > 0, key)
    require(type(record['source_sha']) is str and SHA.fullmatch(record['source_sha']), 'source sha')
    require(type(record['source_digest']) is str and HEX.fullmatch(record['source_digest']), 'source digest')
    require(record['artifact_name'] == 'dist-' + record['source_sha'], 'artifact name')
    require(tag == 'v' + record['version'], 'tag/version mismatch')
    require(type(record['files']) is dict and set(record['files']) == expected_files(record['version']), 'file keys')
    for value in record['files'].values():
        require(type(value) is dict and set(value) == {'sha256', 'size'}, 'file fields')
        require(type(value['sha256']) is str and HEX.fullmatch(value['sha256']), 'file hash')
        require(type(value['size']) is int and value['size'] > 0, 'file size')
    q = record['qualification']
    require(type(q) is dict and set(q) == {'outcome', 'scope', 'report_sha256'}, 'qualification fields')
    require(q['outcome'] == 'pass' and q['scope'] == 'hosted-native-pure-installed', 'qualification outcome/scope')
    require(type(q['report_sha256']) is str and HEX.fullmatch(q['report_sha256']), 'qualification report hash')
    return record


def verify_run(record, run, artifact):
    require(run['id'] == record['run_id'] and run['head_sha'] == record['source_sha'], 'run identity')
    require(run['repository']['full_name'] == REPOSITORY and run['head_repository']['full_name'] == REPOSITORY,
            'run repository')
    require(run['path'] == WORKFLOW and run['event'] in ('push', 'workflow_dispatch'), 'build workflow')
    require(type(run.get('head_branch')) is str and run['head_branch'].startswith('qualify/0.4.0-'),
            'qualification branch')
    require(run['status'] == 'completed' and run['conclusion'] == 'success', 'hosted build not successful')
    require(artifact['id'] == record['artifact_id'] and artifact['name'] == record['artifact_name'], 'artifact identity')
    require(artifact['expired'] is False and artifact['workflow_run']['id'] == record['run_id'], 'artifact run/expiry')
    require(artifact['workflow_run']['head_sha'] == record['source_sha'], 'artifact source')


def verify_bundle(record, source, dist):
    source = Path(source)
    version = tomllib.loads((source / 'pyproject.toml').read_text())['project']['version']
    require(version == record['version'], 'current version')
    require(source_digest(source) == record['source_digest'], 'current source changed')
    require(digest(source / 'release/HOSTED-QUALIFICATION.txt') == record['qualification']['report_sha256'],
            'qualification report changed')
    require(files(dist, version) == record['files'], 'qualified bytes changed')
    spec = importlib.util.spec_from_file_location('release_artifacts', source / 'scripts/release_artifacts.py')
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    release.inspect_bundle(source, Path(dist))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('manifest')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dist', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('record')
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--github-output', type=Path)
    p = sub.add_parser('verify')
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dist', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'manifest':
        version = tomllib.loads((args.source / 'pyproject.toml').read_text())['project']['version']
        body = {'schema': 1, 'repository': os.environ['GITHUB_REPOSITORY'],
                'run_id': int(os.environ['GITHUB_RUN_ID']), 'run_attempt': int(os.environ['GITHUB_RUN_ATTEMPT']),
                'source_sha': os.environ['GITHUB_SHA'], 'source_digest': source_digest(args.source),
                'version': version, 'files': files(args.dist, version),
                'builder': {'python': sys.version, 'platform': platform.platform(),
                            'runner_image': os.environ.get('ImageVersion')}}
        require(body['repository'] == REPOSITORY and SHA.fullmatch(body['source_sha']), 'builder identity')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(body, indent=2) + '\n')
        return
    record = validate_record(json.loads(args.record.read_text()), args.tag)
    if args.command == 'record':
        if args.github_output:
            with args.github_output.open('a') as f:
                for key in ('run_id', 'artifact_id', 'artifact_name'):
                    f.write(key + '=' + str(record[key]) + '\n')
        print(json.dumps(record, indent=2))
    else:
        verify_run(record, json.loads(args.run.read_text()), json.loads(args.artifact.read_text()))
        verify_bundle(record, args.source, args.dist)
        print('Qualified hosted artifact bytes and source binding verified')


if __name__ == '__main__':
    main()
