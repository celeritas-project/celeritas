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

install_hook () {
if [ ! -f "$1" ]; then
  printf "\e[33mCreating post-commit hook at $1.\e[0m\n"
  echo "#!/bin/sh" > "$1"
  chmod a+x "$1"
fi
}

add_hook_scripts () {
  SUFFIX=$1
  HOOK=$2
  BASE=$(basename $HOOK)
  if ! grep $SUFFIX ${HOOK} >/dev/null ; then
    printf "\e[33mAppending ${SUFFIX} call to ${HOOK}\e[0m\n"
    cat >> "${HOOK}" << EOF
GCF="\$(git rev-parse --show-toplevel)/scripts/dev/$BASE.$SUFFIX"
test -x "\${GCF}" && "\${GCF}" "\$@"
EOF
  fi
}

GIT_WORK_TREE="$(git rev-parse --show-toplevel)"
POSTCOMMIT=${GIT_WORK_TREE}/.git/hooks/post-commit
PRECOMMIT=${GIT_WORK_TREE}/.git/hooks/pre-commit

# Ensure a post-commit hook exists (Git LFS might have created one).
install_hook "$POSTCOMMIT"
install_hook "$PRECOMMIT"

printf "\e[2;37mSetting clang format options in git config\e[0m\n"
git config clangFormat.extension "cc,hh,h,cpp,hpp,cu,cuh"
git config clangFormat.style "file"

add_hook_scripts "git-clang-format" "$POSTCOMMIT"
add_hook_scripts "validate-source" "$PRECOMMIT"

printf "\e[0;32mCommit hooks successfully installed for ${GIT_WORK_TREE}\e[0m\n"

###############################################################################
# end of scripts/dev/install-commit-hooks.sh
###############################################################################
