#------------------------------- -*- cmake -*- -------------------------------#
# SPDX-License-Identifier: Apache-2.0
#
# https://github.com/sethrj/cmake-git-version
#
# Copyright 2021-2025 UT-Battelle, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#[=======================================================================[.rst:

CgvFindVersion
--------------

.. command:: cgv_find_version

  Get the project version using Git descriptions to ensure the version numbers
  are always synchronized between Git and CMake::

    cgv_find_version([<projname>])

  ``<projname>``
    Name of the project.

  This command sets the numeric (usable in CMake version comparisons) and
  extended (useful for exact versioning) version variables in the parent
  package::

    ${projname}_VERSION
    ${projname}_VERSION_STRING

  It takes the project name as an optional argument so that it may be used
  *before* calling the CMake ``project`` command.

  The project version string uses an approximation to SemVer strings, appearing
  as v0.1.2 if the version is actually a tagged release, or
  v0.1.3-2+branch.abcdef if it's not. Pre-releases should be tagged as
  v1.0.0-rc.1, and subsequent commits will show as v1.0.0-rc.1.23+branch.abc123.

  If a non-tagged version is exported, or an untagged shallow git clone is used,
  it's impossible to determine the version from the tag, so a warning will be
  issued and the version will be set to 0.0.0.

  The default regex used to match the numeric version and full version string
  from the git tag is::

    v([0-9.]+)(-[a-z]+[0-9.]*)?

  but you can override the regex by setting the ``CGV_TAG_REGEX`` variable
  before calling ``cgv_find_version``. For example, Geant4 tags such as
  ``geant4-11-02-ref-09`` can be matched with::

    geant4-([0-9-]+[0-9]+)(-[a-z]+-[0-9]+)?

  Finally, this script records the time stamp of the file used to generate the
  metadata, and it will re-run cmake if that file changes, and re-run the
  associated git commands only if the file changes.

  .. note:: In order for this script to work properly with archived git
    repositories (generated with ``git-archive`` or GitHub's release tarball
    feature), it's necessary to add to your ``.gitattributes`` file::

      CgvFindVersion.cmake export-subst

    The install script included alongside this file (in the original
    repository) sets this property.

This script can also be run from the command line to determine git repository
data::

  cmake -P cmake/CgvFindVersion.cmake

To print only the version string (to stderr), and from a custom directory::

  cmake -DSOURCE_DIR=. -DONLY=VERSION -P cmake/CgvFindVersion.cmake

#]=======================================================================]

if(CMAKE_SCRIPT_MODE_FILE)
  cmake_minimum_required(VERSION 3.8...3.30)
endif()

set(CGV_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

#-----------------------------------------------------------------------------#
# Get a reproducible timestamp
macro(_cgv_timestamp tsfile tsvar)
  if(EXISTS "${tsfile}")
    file(TIMESTAMP "${tsfile}" ${tsvar} "%Y%m%d.%H%M%S" UTC)
  else()
    set(${tsvar} "")
  endif()
endmacro()

#-----------------------------------------------------------------------------#
# Execute a command, logging verbosely, saving output
macro(_cgv_git_call_output output_var)
  message(VERBOSE "Executing ${GIT_EXECUTABLE} from ${CGV_SOURCE_DIR}: ${ARGV}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${CGV_SOURCE_DIR}"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIT_ERR
    RESULT_VARIABLE GIT_RESULT
    OUTPUT_VARIABLE ${output_var}
  )
endmacro()

#-----------------------------------------------------------------------------#
# Save the version with a timestamp to a cache variable

function(_cgv_store_version vstring vsuffix vhash tsfile)
  if(NOT vstring)
    message(WARNING "The version metadata for ${CGV_PROJECT} could not "
      "be determined: installed version number may be incorrect")
  endif()
  # Replace 11-03 with 11.3
  string(REGEX REPLACE "-+0*" "." vstring "${vstring}")
  # Remove trailing periods
  string(REGEX REPLACE "\\.+$" "" vstring "${vstring}")
  # Remove leading zeros from version components
  string(REGEX REPLACE "0+([1-9]+[0-9]*)" "\\1" vstring "${vstring}")

  # Get timestamp
  _cgv_timestamp("${tsfile}" _vtimestamp)
  # Set up cached data list
  set(_CACHED_VERSION
    "${vstring}" "${vsuffix}" "${vhash}" "${tsfile}" "${_vtimestamp}"
  )
  # Note: extra 'unset' is necessary if using CMake presets with
  # ${CGV_PROJECT}_GIT_DESCRIBE="", even with INTERNAL/FORCE
  unset("${CGV_CACHE_VAR}" CACHE)
  set("${CGV_CACHE_VAR}" "${_CACHED_VERSION}" CACHE INTERNAL
    "Version string and hash for ${CGV_PROJECT}")
  message(VERBOSE "Set ${CGV_CACHE_VAR}=${vstring};${vsuffix};${vhash} from ${tsfile}")
endfunction()

#-----------------------------------------------------------------------------#
# Get the path to the git head used to describe the current repostiory
function(_cgv_git_path resultvar)
  if(GIT_EXECUTABLE)
    _cgv_git_call_output(_TSFILE "rev-parse" "--git-path" "HEAD")
  else()
    set(GIT_RESULT 1)
    set(GIT_ERR "GIT_EXECUTABLE is not defined")
  endif()
  if(GIT_RESULT)
    message(AUTHOR_WARNING "Failed to get path to git head: ${GIT_ERR}")
    set(_TSFILE)
  else()
    get_filename_component(_TSFILE "${_TSFILE}" ABSOLUTE BASE_DIR
      "${CGV_SOURCE_DIR}"
    )
  endif()

  set(${resultvar} "${_TSFILE}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#
# Process description tag: e.g. v0.4.0-2-gc4af497 or v0.4.0 or v2.0.0-rc.2

function(_cgv_try_parse_git_describe version_string branch_string tsfile)
  # Regex groups:
  #  1: primary version (1.2.3)
  #  2: pre-release: dev/alpha/rc annotation (-rc.1)
  #  3: post-tag description (-123-gabcd123)
  #  4: number of commits since (aka distance to) tag (123)
  #  5: commit hash (abcd213)
  set(_DESCR_REGEX "^${CGV_TAG_REGEX}(-([0-9]+)-g([0-9a-f]+))?")
  string(REGEX MATCH "${_DESCR_REGEX}" _MATCH "${version_string}")
  if(NOT _MATCH)
    message(AUTHOR_WARNING
      "Failed to parse description '${version_string}' with regex '${_DESCR_REGEX}'"
    )
    return()
  endif()

  if(NOT CMAKE_MATCH_3)
    # This is a tagged release!
    _cgv_store_version("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "" "${tsfile}")
    return()
  endif()

  if(CMAKE_MATCH_2)
    # After a pre-release, e.g. -rc.1, for SemVer compatibility
    set(_prerelease "${CMAKE_MATCH_2}.${CMAKE_MATCH_4}")
  else()
    # After a release, e.g. -123
    set(_prerelease "-${CMAKE_MATCH_4}")
  endif()

  if(branch_string)
    set(_suffix "${branch_string}.${CMAKE_MATCH_5}")
  else()
    set(_suffix "${CMAKE_MATCH_5}")
  endif()

  # Qualify the version number with the distance-to-tag and hash
  _cgv_store_version(
    "${CMAKE_MATCH_1}" # 1.2.3
    "${_prerelease}" # -rc.2.3, -beta.1, -123
    "${_suffix}" # abcdef
    "${tsfile}" # timestamp file
  )
endfunction()

#-----------------------------------------------------------------------------#

function(_cgv_try_archive_md)
  # Get a possible Git version generated using git-archive (see the
  # .gitattributes file)
  set(_ARCHIVE_DESCR "$Format:%(describe:tags)$")
  set(_ARCHIVE_TAG "$Format:%D$")
  set(_ARCHIVE_HASH "$Format:%h$")
  if(_ARCHIVE_HASH MATCHES "^\\$.*\\$$")
    # Not a git archive
    return()
  endif()

  set(_TSFILE "${CMAKE_CURRENT_LIST_FILE}")
  string(REGEX MATCH "->\\s+(\\S+)" _MATCH "${_ARCHIVE_TAG}")
  if(_MATCH)
    set(_BRANCH "${CMAKE_MATCH_1}")
  else()
    set(_BRANCH)
  endif()

  if(_ARCHIVE_DESCR)
    _cgv_try_parse_git_describe("${_ARCHIVE_DESCR}" "${_BRANCH}" "${_TSFILE}")
    if(${CGV_CACHE_VAR})
      # Successfully parsed description
      return()
    endif()
  endif()

  string(REGEX MATCH "tag: *${CGV_TAG_REGEX}" _MATCH "${_ARCHIVE_TAG}")
  if(_MATCH)
    set(_VERSION "${CMAKE_MATCH_1}")
    set(_SUFFIX "${CMAKE_MATCH_2}")
    set(_HASH)
  else()
    message(AUTHOR_WARNING
      "Could not match a version tag for "
      "git description '${_ARCHIVE_TAG}': perhaps this archive was not "
      "exported from a tagged commit?"
    )
    string(REGEX MATCH " *([0-9a-f]+)" _MATCH "${_ARCHIVE_HASH}")
    if(NOT _MATCH)
      # Could not even find a git hash
      return()
    endif()

    # Found a hash but no version
    set(_VERSION)
    set(_SUFFIX)
    set(_HASH "${CMAKE_MATCH_1}")
  endif()

  _cgv_store_version("${_VERSION}" "${_SUFFIX}" "${_HASH}" "${_TSFILE}")
endfunction()

#-----------------------------------------------------------------------------#
# Try git's 'describe' function
function(_cgv_try_git_describe)
  # First time calling "git describe"
  if(NOT Git_FOUND)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
      message(WARNING "Could not find Git, needed to find the version tag")
      return()
    endif()
  endif()

  if(CGV_TAG_REGEX MATCHES "^\\^?([a-z-]+)")
    set(_match "--match" "${CMAKE_MATCH_1}*")
  else()
    set(_match)
  endif()

  # Load git description
  _cgv_git_call_output(_VERSION_STRING "describe" "--tags" ${_match})
  if(GIT_RESULT)
    message(AUTHOR_WARNING "No suitable git tags found': ${GIT_ERR}")
    return()
  endif()
  if(GIT_ERR)
    message(AUTHOR_WARNING "git describe warned: ${GIT_ERR}")
  endif()
  if(NOT _VERSION_STRING)
    message(AUTHOR_WARNING "Failed to get ${CGV_PROJECT} version from git: "
      "git describe returned an empty string")
    return()
  endif()

  # Get git branch: may fail if detached, leading to empty output, which is
  # the desired behavior
  _cgv_git_call_output(_BRANCH_STRING "symbolic-ref" "--short" "HEAD")

  _cgv_git_path(_TSFILE)
  _cgv_try_parse_git_describe("${_VERSION_STRING}" "${_BRANCH_STRING}" "${_TSFILE}")
endfunction()

#-----------------------------------------------------------------------------#

function(_cgv_try_git_hash)
  if(NOT GIT_EXECUTABLE)
    return()
  endif()
  # Fall back to just getting the hash
  _cgv_git_call_output(_VERSION_HASH "log" "-1" "--format=%h" "HEAD")
  if(_VERSION_HASH_RESULT)
    message(AUTHOR_WARNING "Failed to get current commit hash from git: "
      "${_VERSION_HASH_ERR}")
    return()
  endif()

  _cgv_git_path(_TSFILE)
  _cgv_store_version("" "" "${_VERSION_HASH}" "${_TSFILE}")
endfunction()

function(_cgv_try_all)
  if(${CGV_CACHE_VAR})
    # Previous configure already set the variable: check the timestamp
    list(LENGTH ${CGV_CACHE_VAR} _len)
    if(_len EQUAL 5)
      list(GET ${CGV_CACHE_VAR} 3 _tsfile)
      list(GET ${CGV_CACHE_VAR} 4 _timestamp)
    else()
      message(VERBOSE "Old cache variable ${CGV_CACHE_VAR}: length=${_len}")
      set(_tsfile)
    endif()
    if(_tsfile)
      _cgv_timestamp("${_tsfile}" _curtimestamp)
      if(_timestamp AND _timestamp STREQUAL _curtimestamp)
        message(VERBOSE "Equal time stamp from ${_tsfile}: ${_timestamp}")
        # Time stamp is equal; version doesn't need to be updated
        return()
      else()
        message(VERBOSE
          "Stale timestamp from ${_tsfile}: ${_timestamp} != ${_curtimestamp}"
        )
      endif()
    endif()
    unset(${CGV_CACHE_VAR} CACHE)
  endif()

  _cgv_try_archive_md()
  if(${CGV_CACHE_VAR})
    return()
  endif()

  _cgv_try_git_describe()
  if(${CGV_CACHE_VAR})
    return()
  endif()

  _cgv_try_git_hash()
  if(${CGV_CACHE_VAR})
    return()
  endif()

  # Fallback: no metadata detected
  set(${CGV_CACHE_VAR} "" "-unknown" "")
endfunction()

#-----------------------------------------------------------------------------#

function(cgv_find_version)
  # Set CGV_ variables that are used in embedded macros/functions
  if(ARGC GREATER 0)
    set(CGV_PROJECT "${ARGV0}")
  elseif(NOT CGV_PROJECT)
    if(NOT CMAKE_PROJECT_NAME)
      message(FATAL_ERROR "Project name is not defined")
    endif()
    set(CGV_PROJECT "${CMAKE_PROJECT_NAME}")
  endif()

  if(NOT CGV_TAG_REGEX)
    set(CGV_TAG_REGEX "v([0-9.]+)(-[a-z]+[0-9.]*)?")
  endif()

  set(CGV_CACHE_VAR "${CGV_PROJECT}_GIT_DESCRIBE")

  # Try all possible ways of obtaining metadata
  _cgv_try_all()

  # Unpack stored version
  set(_CACHED_VERSION "${${CGV_CACHE_VAR}}")
  list(GET _CACHED_VERSION 0 _VERSION_STRING)
  list(GET _CACHED_VERSION 1 _VERSION_STRING_SUFFIX)
  list(GET _CACHED_VERSION 2 _VERSION_HASH)
  list(GET _CACHED_VERSION 3 _TSFILE)

  if(NOT _VERSION_STRING)
    set(_VERSION_STRING "0.0.0")
  endif()

  if(_VERSION_HASH)
    set(_FULL_VERSION_STRING "${_VERSION_STRING}${_VERSION_STRING_SUFFIX}+${_VERSION_HASH}")
  else()
    set(_FULL_VERSION_STRING "${_VERSION_STRING}${_VERSION_STRING_SUFFIX}")
  endif()

  if(_TSFILE)
    # Re-run cmake if the timestamp file changes
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_TSFILE}")
  endif()

  # Set version number and descriptive version in parent scope
  set(${CGV_PROJECT}_VERSION "${_VERSION_STRING}" PARENT_SCOPE)
  set(${CGV_PROJECT}_VERSION_STRING "${_FULL_VERSION_STRING}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------#

if(CMAKE_SCRIPT_MODE_FILE)
  if(DEFINED SOURCE_DIR)
    set(CGV_SOURCE_DIR ${SOURCE_DIR})
  endif()
  cgv_find_version(TEMP)
  if(DEFINED ONLY)
    # Print only the given variable, presumably VERSION or VERSION_STRING
    # (will print to stderr)
    set(VERSION "${TEMP_VERSION}")
    set(VERSION_STRING "${TEMP_VERSION_STRING}")
    message("${${ONLY}}")
  else()
    message("VERSION=\"${TEMP_VERSION}\"")
    message("VERSION_STRING=\"${TEMP_VERSION_STRING}\"")
  endif()
endif()

# cmake-git-version 1.2.1-4+main.86aa5ba
