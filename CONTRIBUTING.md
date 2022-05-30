# Contribution guide

Contributions are welcome!

## Instructions

1. Fork the repo to your own GitHub account

2. Clone the repo and install according to the development instructions. Replace the URL For the repo on `nel-lab` to the fork on your account: https://github.com/nel-lab/mesmerize-core#for-development

3. Checkout the `dev` branch, and then checkout your feature or bug fix branch:

```bash
cd mesmerize-core
git checkout dev
git checkout -b my-new-feature-branch
# make some changes, lint with black, and commit
black .
git add my_changed_files
git commit -m "my new feature"
git push origin my-new-feature-branch
```

4. Run tests, make sure they pass

```bash
cd mesmerize-core
DOWNLOAD_GROUND_TRUTHS=1 pytest -s .
```

5. Finally make a PR against the `dev` branch, the PR will also run tests using our CI pipeline to make sure tests pass. We will get back to your with any further suggestions!
