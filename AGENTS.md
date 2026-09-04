# Adding a dataset entry

Add new datasets to `README.md` under the most specific existing heading for the imagery source or task. Before adding an entry, search the README for the dataset name, GitHub organization, and URL to avoid duplicates.

Use a Markdown bullet that follows the existing style:

```markdown
* [Dataset name](https://github.com/owner/repository) -> Short, factual description of the dataset. Paper: Full Paper Title
```

Include:

- The official dataset name.
- The direct GitHub repository URL, when one exists. Do not link to a GitHub search or user profile.
- The full paper name as plain text, when a paper exists. Do not link paper, DOI, or arXiv pages because these links often break CI link checks.
- A concise description covering the imagery source or sensor, task, labels, scale, resolution, geography, or temporal coverage when that information is useful.

If the dataset page is not GitHub, use its official download or project page as the main link and add the GitHub repository separately:

```markdown
* [Dataset name](https://official.dataset/page) -> [GitHub](https://github.com/owner/repository). Short, factual description. Paper: Full Paper Title
```

If no paper or GitHub repository exists, omit that part rather than guessing. Preserve the surrounding section's bullet marker and wording style, keep the entry on one line where practical, and do not reorder or reformat unrelated entries.

After editing, verify that every new link is well formed and review the diff for accidental changes.
