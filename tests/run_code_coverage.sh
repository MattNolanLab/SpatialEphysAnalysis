CODECOV_TOKEN=$1
CI_COMMIT_SHA=$2
conda activate env && pip install pytest-cov codecov && python3 -m pytest --cov-report xml --cov-report term --cov=./ tests/unit && python3 -m codecov -t ${CODECOV_TOKEN} --commit=${CI_COMMIT_SHA} --file=coverage.xml
