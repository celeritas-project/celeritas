//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/SenseUtils.hh
//! \brief Sense helper functions and types.
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/Bitset.hh"

#include "OrangeTypes.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST/DEVICE)
//---------------------------------------------------------------------------//
/*!
 * Convert a boolean value to a Sense enum.
 */
CELER_CONSTEXPR_FUNCTION Sense to_sense(bool s)
{
    return static_cast<Sense>(s);
}

//---------------------------------------------------------------------------//
/*!
 * Change the sense across a surface.
 */
[[nodiscard]] CELER_CONSTEXPR_FUNCTION Sense flip_sense(Sense orig)
{
    return static_cast<Sense>(!static_cast<bool>(orig));
}

//---------------------------------------------------------------------------//
/*!
 * Change the sense across a surface.
 */
[[nodiscard]] CELER_CONSTEXPR_FUNCTION SignedSense flip_sense(SignedSense orig)
{
    using IntT = std::underlying_type_t<SignedSense>;
    return static_cast<SignedSense>(-static_cast<IntT>(orig));
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the sense based on the LHS expression of the quadric equation.
 *
 * This is an optimized jump-free version of:
 * \code
    return quadric == 0 ? SignedSense::on
        : quadric < 0 ? SignedSense::inside
        : SignedSense::outside;
 * \endcode
 * as
 * \code
    int gz = !(quadric <= 0) ? 1 : 0;
    int lz = quadric < 0 ? 1 : 0;
    return static_cast<SignedSense>(gz - lz);
 * \endcode
 * and compressed into a single line.
 *
 * NaN values are treated as "outside".
 */
[[nodiscard]] CELER_CONSTEXPR_FUNCTION SignedSense
real_to_sense(real_type quadric)
{
    return static_cast<SignedSense>(!(quadric <= 0) - (quadric < 0));
}

//---------------------------------------------------------------------------//
/*!
 * Convert a signed sense to a Sense enum.
 */
CELER_CONSTEXPR_FUNCTION Sense to_sense(SignedSense s)
{
    return Sense(static_cast<int>(s) >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a signed sense to a surface state.
 */
CELER_CONSTEXPR_FUNCTION SurfaceState to_surface_state(SignedSense s)
{
    return s == SignedSense::on ? SurfaceState::on : SurfaceState::off;
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS (HOST)
//---------------------------------------------------------------------------//
//! Get a printable character corresponding to a sense.
inline constexpr char to_char(Sense s)
{
    return s == Sense::inside ? '-' : '+';
}

// Get a string corresponding to a signed sense
inline char const* to_cstring(SignedSense s)
{
    switch (s)
    {
        case SignedSense::inside:
            return "inside";
        case SignedSense::on:
            return "on";
        case SignedSense::outside:
            return "outside";
    }
    return "<invalid>";
}

//---------------------------------------------------------------------------//
// CLASSES
//---------------------------------------------------------------------------//
/*!
 * Wrapper for a sense value that is optionally set.
 */
class SenseValue
{
  private:
    enum : char
    {
        sense_bit,
        is_assigned_bit,
    };

  public:
    constexpr SenseValue() = default;

    //! Construct with a sense value
    CELER_CONSTEXPR_FUNCTION SenseValue(Sense sense)
    {
        sense_[sense_bit] = static_cast<bool>(sense);
        sense_[is_assigned_bit] = true;
    }

    //! Convert to a sense value
    CELER_CONSTEXPR_FUNCTION operator Sense() const
    {
        return to_sense(sense_[sense_bit]);
    }

    //! Convert to a boolean value
    CELER_CONSTEXPR_FUNCTION explicit operator bool() const
    {
        return sense_[sense_bit];
    }

    //! Assign a sense value
    CELER_CONSTEXPR_FUNCTION SenseValue& operator=(Sense sense)
    {
        sense_[sense_bit] = static_cast<bool>(sense);
        sense_[is_assigned_bit] = true;
        return *this;
    }

    //! Check whether there is a cached sense value
    CELER_CONSTEXPR_FUNCTION bool is_assigned() const
    {
        return sense_[is_assigned_bit];
    }

    //! Clear the sense value
    CELER_CONSTEXPR_FUNCTION void reset() { sense_.reset(); }

  private:
    Bitset<2> sense_;
};

}  // namespace celeritas
