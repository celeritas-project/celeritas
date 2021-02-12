#---------------------------------*-CMake-*----------------------------------#
# SPDX-License-Identifier: Apache-2.0
#
# https://github.com/sethrj/cmake-git-version
#
#  Copyright 2021 UT-Battelle, LLC and Seth R Johnson
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#[=======================================================================[.rst:

CgvFindVersion
--------------

.. command:: cgv_find_version

  Get the project version using Git descriptions to ensure the version numbers
  are always synchronized between Git and CMake::

    cgv_find_version([<projname>])

  ``<projname>``
    Name of the project.

  This command sets the following variables in the parent package::

    ${projname}_VERSION
    ${projname}_VERSION_STRING

  It takes the project name and version file path as optional arguments to
  support using it before the CMake ``project`` command.

  The project version string uses an approximation to SemVer strings, appearing
  as v0.1.2 if the version is actually a tagged release, or v0.1.3+abcdef if
  it's not.

  If a non-tagged version is exported, or an untagged shallow git clone is used,
  it's impossible to determine the version from the tag, so a warning will be
  issued and the version will be set to 0.0.0.

  The exact regex used to match the version tag is::

    v([0-9.]+)(-dev[0-9.]+)?


  .. note:: In order for this script to work properly with archived git
    repositories (generated with ``git-archive`` or GitHub's release tarball
    feature), it's necessary to add to your ``.gitattributes`` file::

      CgvFindVersion.cmake export-subst

#]=======================================================================]

if(CMAKE_SCRIPT_MODE_FILE)
  cmake_minimum_required(VERSION 3.8)
endif()

function(cgv_find_version)
  set(projname "${ARGV0}")
  if(NOT projname)
    set(projname "${CMAKE_PROJECT_NAME}")
    if(NOT projname)
      message(FATAL_ERROR "Project name is not defined")
    endif()
  endif()

  # Get a possible Git version generated using git-archive (see the
  # .gitattributes file)
  set(_ARCHIVE_TAG "$Format:%D$")
  set(_ARCHIVE_HASH "$Format:%h$")

  set(_TAG_REGEX "v([0-9.]+)(-dev[0-9.]+)?")
  set(_HASH_REGEX "([0-9a-f]+)")

  if(_ARCHIVE_HASH MATCHES "%h")
    # Not a "git archive": use live git information
    set(_CACHE_VAR "${projname}_GIT_DESCRIBE")
    set(_CACHED_VERSION "${${_CACHE_VAR}}")
    if(NOT _CACHED_VERSION)
      # Building from a git checkout rather than a distribution
      if(NOT GIT_EXECUTABLE)
        find_package(Git QUIET REQUIRED)
      endif()
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" "describe" "--tags" "--match" "v*"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        ERROR_VARIABLE _GIT_ERR
        OUTPUT_VARIABLE _VERSION_STRING
        RESULT_VARIABLE _GIT_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(_GIT_RESULT)
        message(AUTHOR_WARNING "No git tags in ${projname} matched 'v*': "
          "${_GIT_ERR}")
      elseif(NOT _VERSION_STRING)
        message(WARNING "Failed to get ${projname} version from git: "
          "git describe returned an empty string")
      else()
        # Process description tag: e.g. v0.4.0-2-gc4af497 or v0.4.0
        # or v2.0.0-dev2
        string(REGEX MATCH "^${_TAG_REGEX}(-[0-9]+-g${_HASH_REGEX})?" _MATCH
          "${_VERSION_STRING}"
        )
        if(_MATCH)
          set(_VERSION_STRING "${CMAKE_MATCH_1}")
          set(_VERSION_STRING_SUFFIX "${CMAKE_MATCH_2}")
          if(CMAKE_MATCH_3)
            # *not* a tagged release
            set(_VERSION_HASH "${CMAKE_MATCH_4}")
          endif()
        endif()
      endif()
      if(NOT _VERSION_STRING)
        execute_process(
          COMMAND "${GIT_EXECUTABLE}" "log" "-1" "--format=%h" "HEAD"
          WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
          OUTPUT_VARIABLE _VERSION_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
      endif()
      set(_CACHED_VERSION "${_VERSION_STRING}" "${_VERSION_STRING_SUFFIX}" "${_VERSION_HASH}")
      set("${_CACHE_VAR}" "${_CACHED_VERSION}" CACHE INTERNAL
        "Version string and hash for ${projname}")
    endif()
    list(GET _CACHED_VERSION 0 _VERSION_STRING)
    list(GET _CACHED_VERSION 1 _VERSION_STRING_SUFFIX)
    list(GET _CACHED_VERSION 2 _VERSION_HASH)
  else()
    string(REGEX MATCH "tag: *${_TAG_REGEX}" _MATCH "${_ARCHIVE_TAG}")
    if(_MATCH)
      set(_VERSION_STRING "${CMAKE_MATCH_1}")
      set(_VERSION_STRING_SUFFIX "${CMAKE_MATCH_2}")
    else()
      message(AUTHOR_WARNING "Could not match a version tag for "
        "git description '${_ARCHIVE_TAG}': perhaps this archive was not "
        "exported from a tagged commit?")
      string(REGEX MATCH " *${_HASH_REGEX}" _MATCH "${_ARCHIVE_HASH}")
      if(_MATCH)
        set(_VERSION_HASH "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()

  if(NOT _VERSION_STRING)
    set(_VERSION_STRING "0.0.0")
  endif()

  if(_VERSION_HASH)
    set(_FULL_VERSION_STRING "v${_VERSION_STRING}${_VERSION_STRING_SUFFIX}+${_VERSION_HASH}")
  else()
    set(_FULL_VERSION_STRING "v${_VERSION_STRING}${_VERSION_STRING_SUFFIX}")
  endif()

  set(${projname}_VERSION "${_VERSION_STRING}" PARENT_SCOPE)
  set(${projname}_VERSION_STRING "${_FULL_VERSION_STRING}" PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
  cgv_find_version(TEMP)
  message("VERSION=\"${TEMP_VERSION}\"")
  message("VERSION_STRING=\"${TEMP_VERSION_STRING}\"")
endif()
