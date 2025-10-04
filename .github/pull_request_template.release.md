<!--
Title: "Release vX.Y.Z"
Target: upstream/develop (always, even if backporting!)
-->

**Release branches must be named `release-vX.Y.Z`**

## Pre-merge checklist

- [ ] Ensure all CI jobs on the branch to be released (develop or backports/vX.Y) pass.
- [ ] If releasing from develop, tag the develop branch with ``vX.Y.Z-rc.N`` where N starts with 1, and increment for every time you return to this step due to new pull requests.
- [ ] Run performance regression tests on Perlmutter, Frontier, and an additional machine with debug assertions enabled (e.g., Wildstyle).
- [ ] Update documentation with release notes from all pull requests newly included in the release.
- [ ] Ensure PDF documentation builds without error.

### Major release only

- [ ] Check for (and delete if found) code marked as "deprecated: to be removed in vX".
- [ ] Update the zenodo reference

## Post-merge checklist

- [ ] If releasing a backported version branch, cherry-pick this documentation commit into the backport branch.
- [ ] Use the [release scripts](https://github.com/celeritas-project/release-scripts) to:
  - Add a text description of the release based on the change set
  - Draft a new GitHub release
  - (then, using the GitHub web interface) Publish the release
  - Draft a zenodo release
  - (then, using the Zenodo web interface) Publish the zenodo citation

## Post-release checklist

- [ ] Use the release script to update the archive.
- [ ] Pull locally (make sure to use the ``--tags`` option) and build PDF user documentation for the release, and upload it. Ensure breathe is activated (so the API is listed) and that the version is embedded correctly.
- [ ] Update the Spack recipe for Celeritas with the new version and the sha256 value returned by the release script. Submit a [pull request to the Spack project](https://github.com/spack/spack/pull).
- [ ] Mark the GitHub [release milestone](https://github.com/celeritas-project/celeritas/milestones) as completed.
