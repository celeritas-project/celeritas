//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Version.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>
#include <cstdlib>  // IWYU pragma: keep
#include <iosfwd>
#include <string>
#include <string_view>

// Undefine macros from sys/sysmacros.h
#ifdef major
#    undef major
#endif
#ifdef minor
#    undef minor
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple version comparison.
 *
 * This is intended to be constructed with version numbers from \c
 * celeritas_cmake_strings.hh and used in unit tests. It can be used in `if
 * constexpr` expressions with preprocessor macros. In the constructor
 * documentation, x/y/z correspond to major/minor/patch.
 *
 * \code
 * Version(4) == Version(4.0) == Version(4.0.0)
 * Version(3.1) > Version(3)
 * \endcode
 */
class Version
{
  public:
    //!@{
    //! \name Type aliases
    using size_type = unsigned int;
    using ArrayT = std::array<size_type, 3>;
    //!@}

  public:
    // Construct from a string "1.2.3"
    static Version from_string(std::string_view sv);

    // Construct from an 0xXXYYZZ integer
    static inline constexpr Version from_hex_xxyyzz(size_type value);

    // Construct from an 0xXXYZ integer
    static inline constexpr Version from_dec_xyz(size_type value);

    // Construct from x.y.z integers
    inline constexpr Version(size_type major,
                             size_type minor = 0,
                             size_type patch = 0);

    //!@{
    //! \name Accessors

    //! Get version as an array
    constexpr ArrayT const& value() const { return version_; }

    //! Get major version
    constexpr size_type major() const { return version_[0]; }

    //! Get minor version
    constexpr size_type minor() const { return version_[1]; }

    //! Get patch version
    constexpr size_type patch() const { return version_[2]; }

    //!@}

  private:
    ArrayT version_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from an 0xXXYYZZ integer.
 *
 * This version scheme is used by Celeritas. (The leading 0x prevents
 * version `01` from turning the expression into an octal.)
 */
constexpr Version Version::from_hex_xxyyzz(size_type value)
{
    return {(value >> 16 & 0xffu), (value >> 8 & 0xffu), (value & 0xffu)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a decimal with XYZ or XXYZ.
 *
 * This version scheme is used by Geant4.
 */
constexpr Version Version::from_dec_xyz(size_type value)
{
    return {(value / 100), (value / 10) % 10, value % 10};
}

//---------------------------------------------------------------------------//
/*!
 * Construct from x.y.z integers.
 */
constexpr Version::Version(size_type major, size_type minor, size_type patch)
    : version_{{major, minor, patch}}
{
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// NOTE: constexpr is only defined for std::array in C++20

#define CELER_DEFINE_VERSION_CMP(TOKEN)                                \
    inline bool operator TOKEN(Version const& lhs, Version const& rhs) \
    {                                                                  \
        return lhs.value() TOKEN rhs.value();                          \
    }

CELER_DEFINE_VERSION_CMP(==)
CELER_DEFINE_VERSION_CMP(!=)
CELER_DEFINE_VERSION_CMP(<)
CELER_DEFINE_VERSION_CMP(>)
CELER_DEFINE_VERSION_CMP(<=)
CELER_DEFINE_VERSION_CMP(>=)

#undef CELER_DEFINE_VERSION_CMP

// Write to stream
std::ostream& operator<<(std::ostream&, Version const&);

// Get the Celeritas version as an object
Version celer_version();

//---------------------------------------------------------------------------//
}  // namespace celeritas
