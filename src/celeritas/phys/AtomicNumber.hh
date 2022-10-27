//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/AtomicNumber.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe atomic number identifier.
 *
 * Atomic numbers have constraints that integers do not (zero is not an atomic
 * number, dividing one atomic number by another is meaningless) so this type
 * improves the semantics and safety of Z.
 *
 * Generally speaking, Z numbers are used during setup and when evaluating
 * expressions that use atomic charge. Z should not be used to "index" data
 * during runtime.
 *
 * Constructing with a nonpositive Z gives the result a "false" state (\c get
 * will throw a \c DebugError).
 */
class AtomicNumber
{
  public:
    //! Construct with an invalid/unassigned value of zero
    CELER_CONSTEXPR_FUNCTION AtomicNumber() = default;

    //! Construct with the Z value
    explicit CELER_CONSTEXPR_FUNCTION AtomicNumber(int z_value) : z_(z_value)
    {
    }

    //! True if value is assigned/valid
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const { return z_ > 0; }

    //! Get the Z value
    CELER_CONSTEXPR_FUNCTION int unchecked_get() const { return z_; }

    // Get the Z value
    inline CELER_FUNCTION int get() const;

  private:
    int z_{0};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the Z value.
 */
inline CELER_FUNCTION int AtomicNumber::get() const
{
    CELER_ENSURE(*this);
    return z_;
}

//---------------------------------------------------------------------------//
// COMPARATORS
//---------------------------------------------------------------------------//
#define CELER_DEFINE_ATOMICNUMBER_CMP(TOKEN)                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(AtomicNumber lhs, \
                                                 AtomicNumber rhs) \
    {                                                              \
        return lhs.unchecked_get() TOKEN rhs.unchecked_get();      \
    }

CELER_DEFINE_ATOMICNUMBER_CMP(==)
CELER_DEFINE_ATOMICNUMBER_CMP(!=)
CELER_DEFINE_ATOMICNUMBER_CMP(<)
CELER_DEFINE_ATOMICNUMBER_CMP(>)
CELER_DEFINE_ATOMICNUMBER_CMP(<=)
CELER_DEFINE_ATOMICNUMBER_CMP(>=)

#undef CELER_DEFINE_ATOMICNUMBER_CMP

//---------------------------------------------------------------------------//
} // namespace celeritas
