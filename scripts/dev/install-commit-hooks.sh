#!/bin/sh -e
###############################################################################
# File  : scripts/dev/install-commit-hooks.sh
#
# Install a script to run git-clang-format after each commit.
###############################################################################

if ! hash git-clang-format ; then
  printf "\e[31mgit-clang-format is not installed.\e[0m
Install clang-format and update your paths.
"
  exit 1
fi

GIT_WORK_TREE="$(git rev-parse --show-toplevel)"
POSTCOMMIT=${GIT_WORK_TREE}/.git/hooks/post-commit

# Ensure a post-commit hook exists (Git LFS might have created one).
if [ ! -f "${POSTCOMMIT}" ]; then
  printf "\e[33mCreating post-commit hook at ${POSTCOMMIT}.\e[0m\n"
  echo "#!/bin/sh" > "${POSTCOMMIT}"
  chmod a+x "${POSTCOMMIT}"
fi

printf "\e[2;37mSetting clang format options in git config\e[0m\n"
git config clangFormat.extension "cc,hh,h,cpp,hpp,cu,cuh"
git config clangFormat.style "file"

if ! grep 'git-clang-format' ${POSTCOMMIT} >/dev/null ; then
  printf "\e[33mAppending git-clang-format call to ${POSTCOMMIT}\e[0m\n"
  cat >> "${POSTCOMMIT}" << 'EOF'
GCF="$(git rev-parse --show-toplevel)/scripts/dev/post-commit.git-clang-format"
test -x "${GCF}" && "${GCF}" "$@"
EOF
fi

printf "\e[0;32mPre-commit hook successfully installed for ${GIT_WORK_TREE}\e[0m\n"

###############################################################################
# end of scripts/dev/install-commit-hooks.sh
###############################################################################
