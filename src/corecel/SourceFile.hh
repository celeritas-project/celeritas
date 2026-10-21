//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/SourceFile.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <utility>

#include "corecel/Config.hh"

#include "Macros.hh"

namespace celeritas
{
namespace detail
{
// Internal linkage lets the compiler address the trimmed strings relatively
// instead of exporting one weak symbol per source file
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Storage for a source path copied character-by-character at compile time.
 *
 * Because the characters are template arguments, \c value is a distinct
 * static array rather than a pointer into the \c __FILE__ literal: the
 * compiler never needs to emit the original, longer string.
 */
template<char... Cs>
struct SourceFileChars
{
    static constexpr char value[sizeof...(Cs) + 1] = {Cs..., '\0'};
};

//---------------------------------------------------------------------------//
//! Length of a null-terminated string
constexpr std::size_t cstr_length(char const* s)
{
    std::size_t n = 0;
    while (s[n] != '\0')
    {
        ++n;
    }
    return n;
}

//---------------------------------------------------------------------------//
//! Whether a character separates path components on POSIX or Windows
constexpr bool is_path_separator(char c)
{
    return c == '/' || c == '\\';
}

//---------------------------------------------------------------------------//
//! Whether the path component starting at \c pos is exactly \c dir
constexpr bool is_dir_component(
    char const* s, std::size_t len, std::size_t pos, char const* dir)
{
    std::size_t i = 0;
    for (; dir[i] != '\0'; ++i)
    {
        if (pos + i >= len || s[pos + i] != dir[i])
        {
            return false;
        }
    }
    return pos + i < len && is_path_separator(s[pos + i]);
}

//---------------------------------------------------------------------------//
/*!
 * Index just past the last \c src/, \c app/, or \c test/ path component.
 *
 * Returns zero if none is present, so that paths outside the Celeritas
 * source tree (e.g. downstream code using Celeritas assertions) are kept
 * intact.
 */
constexpr std::size_t source_dir_offset(char const* s)
{
    constexpr char const* dirs[] = {"src", "app", "test"};
    std::size_t const len = cstr_length(s);
    std::size_t result = 0;
    for (std::size_t pos = 0; pos < len; ++pos)
    {
        if (pos != 0 && !is_path_separator(s[pos - 1]))
        {
            continue;
        }
        for (char const* dir : dirs)
        {
            if (is_dir_component(s, len, pos, dir))
            {
                result = pos + cstr_length(dir) + 1;
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Normalize a path character to a forward slash
constexpr char normalize_separator(char c)
{
    return c == '\\' ? '/' : c;
}

//---------------------------------------------------------------------------//
//! Expand the trimmed path into a character pack
template<class S, std::size_t Off, std::size_t... Is>
constexpr char const* make_source_file(std::index_sequence<Is...>)
{
    return SourceFileChars<normalize_separator(S::str()[Off + Is])...>::value;
}

//---------------------------------------------------------------------------//
//! Trim the path returned by \c S::str() at compile time
template<class S>
constexpr char const* source_file()
{
    constexpr std::size_t off = source_dir_offset(S::str());
    constexpr std::size_t len = cstr_length(S::str()) - off;
    return make_source_file<S, off>(std::make_index_sequence<len>{});
}

//---------------------------------------------------------------------------//
}  // namespace
}  // namespace detail
}  // namespace celeritas

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * \def CELER_SOURCE_FILE
 *
 * Expand to the current source file path relative to the Celeritas source
 * tree, as a null-terminated string that lives for the duration of the
 * program.
 *
 * This replaces \c __FILE__ in log and error provenance so that build
 * directories are not compiled into the binary and so that messages show a
 * short path such as <code>corecel/io/Logger.hh</code> rather than a full
 * one. The trimming is done at compile time (see \c
 * celeritas::detail::source_file) and requires only C++17. Paths that do not
 * contain a \c src/, \c app/, or \c test/ component are left unchanged, and
 * Windows path separators are replaced with forward slashes.
 *
 * Each use expands to a lambda holding a local class, which costs the
 * compiler roughly as much as a small function definition. Trimming is
 * therefore enabled only through the \c CELERITAS_TRIM_SOURCE_FILE
 * configuration option, which is available only when debug assertions are
 * disabled. Otherwise, and always in device code (where the trimmed string
 * would be a host address), this macro is simply \c __FILE__ .
 *
 * \note GCC 8 fails to deduce the lambda's return type inside a discarded
 * <code>if constexpr</code> branch, so trimming is also disabled there.
 */
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 9
#    define CELER_SOURCE_FILE_TRIM_ 0
#else
#    define CELER_SOURCE_FILE_TRIM_ CELERITAS_TRIM_SOURCE_FILE
#endif

#if CELER_SOURCE_FILE_TRIM_ && !CELER_DEVICE_COMPILE
#    define CELER_SOURCE_FILE \
        ([] { \
            struct CelerSourceFile_ \
            { \
                static constexpr char const* str() { return __FILE__; } \
            }; \
            return ::celeritas::detail::source_file<CelerSourceFile_>(); \
        }())
#else
#    define CELER_SOURCE_FILE __FILE__
#endif

//---------------------------------------------------------------------------//
