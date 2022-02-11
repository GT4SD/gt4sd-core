# Contributing

<!-- add missing CLA -->

## Contributing to GT4SD codebase

If you would like to contribute to the package, we recommend the following development setup.

1. Create a copy of the [repository](https://github.com/GT4SD/gt4sd-core) via the "_Fork_" button.

2. Clone the gt4sd-core repository:

    ```sh
    git clone git@github.com:${GH_ACCOUNT_OR_ORG}/gt4sd-core.git
    ```

3. Create a dedicated branch:

    ```sh
    cd gt4sd-core
    git checkout -b a-super-nice-feature-we-all-need
    ```

4. Create and activate a dedicated conda environment:

    ```sh
    conda env create -f conda.yml
    conda activate gt4sd
    ```

5. Install `gt4sd` in editable mode:

    ```sh
    pip install -e.
    ```

6. Implement your changes and once you are ready run the tests:

    ```sh
    python -m pytest -sv
    ```

    And the style checks:

    ```sh
    # blacking and sorting imports
    python -m black src/gt4sd
    python -m isort src/gt4sd
    # checking flake8 and mypy
    python -m flake8 --disable-noqa --per-file-ignores="__init__.py:F401" src/gt4sd
    python -m mypy src/gt4sd
    ```

7. Once the tests and checks passes, but most importantly you are happy with the implemented feature commit your changes.

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

8. From your fork, open a pull request via the "_Contribute_" button, the maintainers will be happy to review it.

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
