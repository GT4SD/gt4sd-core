# Contributing

## Contribution Terms and License

The documentation and benchmarking of GT4SD is contained in this repository. To contribute
to this project or any of the elements of GT4SD we recommend you start by reading this
contributing guide.

## Contributor License Agreement

Before you can submit any code we need all contributors to sign a
contributor license agreement. By signing a contributor license
agreement (CLA) you're basically just attesting to the fact
that you are the author of the contribution and that you're freely
contributing it under the terms of the MIT License.

When you contribute to the GT4SD project with a new pull request,
a bot will evaluate whether you have signed the CLA. If required, the
bot will comment on the pull request, including a link to accept the
agreement. The [individual CLA](./iCLA.md) document is available for review in this repo.
You can sign the agreement right away by opening an issue clicking [here](https://github.com/GT4SD/gt4sd-core/issues/new?assignees=&labels=cla-signing&template=cla-signature.yaml&title=CLA+signature).

## Contributing to GT4SD codebase

If you would like to contribute to the package, we recommend the following development setup.

1. Create a copy of the [repository](https://github.com/GT4SD/gt4sd-core) via the "_Fork_" button.

2. Clone the gt4sd-core repository:

    ```sh
    git clone git@github.com:${GH_ACCOUNT_OR_ORG}/gt4sd-core.git
    ```

3. Add remote gt4sd-core repo as an "upstream" in your local repo, so you can check/update remote changes.

   ```sh
   git remote add upstream git@github.com:GT4SD/gt4sd-core.git
   ```

4. Create a dedicated branch:

    ```sh
    cd gt4sd-core
    git checkout -b a-super-nice-feature-we-all-need
    ```

5. Create and activate a dedicated conda environment:

    ```sh
    conda env create -f conda_cpu_mac.yml # for linux use conda_cpu_linux.yml
    conda activate gt4sd
    ```

6. Install `gt4sd` in editable mode:

    ```sh
    pip install -e.
    ```

7. Implement your changes and once you are ready run the tests:

    ```sh
    # this can take quite long a it's downloading all models and running them multiple times in the tests
    python -m pytest -sv
    ```

    And the style checks:

    ```sh
    # blacking and sorting imports (this might change your files)
    python -m black src/gt4sd
    # checking flake8 and mypy
    python -m flake8 --disable-noqa --per-file-ignores="__init__.py:F401" src/gt4sd
    python -m mypy src/gt4sd
    ```

    Ensure the license headers:

    ```sh
    licenseheaders -y 2023 -d src/gt4sd -o "GT4SD team" -t mit.tmpl
    ```

8. Once the tests and checks passes, but most importantly you are happy with the implemented feature, commit your changes.

    ```sh
    # add the changes
    git add 
    # commit them
    git commit -s -m "feat: implementing super nice feature." -m "A feature we all need."
    # check upstream changes
    git fetch upstream
    git rebase upstream/main
    # push changes to your fork
    git push -u origin a-super-nice-feature-we-all-need
    ```

9. From your fork, open a pull request via the "_Contribute_" button, the maintainers will be happy to review it.

## Contributing to GT4SD documentation

We recommend the "Python Docstring Generator" extension in VSCode.

However, the types should not be duplicated.
The sphinx documentation will pick it up from [type annotations](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#type-annotations).
Unfortunately, a custom template is required to not add any types at all.

Its settings are:

```json
    "autoDocstring.docstringFormat": "google",
    "autoDocstring.startOnNewLine": false,
    "autoDocstring.customTemplatePath": "/absolute_path_to/.google_pep484.mustache"
```

where the last line would point to the custom template file (e.g. in your user home)
with the following content: (just placeholders for types are removed):

```tpl
{{! Google Docstring Template }}
{{summaryPlaceholder}}

{{extendedSummaryPlaceholder}}

{{#parametersExist}}
Args:
{{#args}}
    {{var}}: {{descriptionPlaceholder}}
{{/args}}
{{#kwargs}}
    {{var}}: {{descriptionPlaceholder}}. Defaults to {{&default}}.
{{/kwargs}}
{{/parametersExist}}

{{#exceptionsExist}}
Raises:
{{#exceptions}}
    {{type}}: {{descriptionPlaceholder}}
{{/exceptions}}
{{/exceptionsExist}}

{{#returnsExist}}
Returns:
{{#returns}}
    {{descriptionPlaceholder}}
{{/returns}}
{{/returnsExist}}

{{#yieldsExist}}
Yields:
{{#yields}}
    {{descriptionPlaceholder}}
{{/yields}}
{{/yieldsExist}}
```
