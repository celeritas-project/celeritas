<!--
Title: "Release vX.Y.Z"
Target: upstream/develop (always!)
-->

**Release branches must be named `release-vX.Y.Z`**

## Pre-merge checklist

- [ ] Ensure all CI jobs on develop pass
- [ ] Tag the develop branch with ``vX.Y.Z-rc.N`` where N starts with 1, and increment for every time you return to this step due to new pull requests.
- [ ] Run performance regression tests on Summit, Crusher/Frontier, and an additional machine with debug assertions enabled (e.g., Wildstyle).
- [ ] Update documentation with release notes from all pull requests newly included in the release.
- [ ] Check for (and delete if found) code marked as "deprecated: to be removed in vX.Y".
- [ ] Ensure the code documentation builds with as few warnings as possible in the `doc` workflow on the CI.

## Post-merge checklist

- [ ] If releasing a backported version branch, cherry-pick this documentation commit into the backport branch.
- [ ] Use the [GitHub interface](https://github.com/celeritas-project/celeritas/releases/new) to create a new release with the documentation update that was just added.

## Post-release checklist

- [ ] Save the ``tar.gz`` and attach to the release, because the hash changes if the git "describe" function returns a different result for the release tag's hash (e.g., if a collaborative branch on the main repository points to that commit).
- [ ] Pull locally (make sure to use the ``--tags`` option) and build PDF user documentation for the release. Ensure breathe is activated (so the API is listed) and that the version is embedded correctly.
- [ ] Update the Spack recipe for Celeritas with the new version and sha256 value (either manually or using ``spack checksum``) and submit a [pull request to the Spack project](https://github.com/spack/spack/pull).
- [ ] Mark the GitHub [release milestone](https://github.com/celeritas-project/celeritas/milestones) as completed.
