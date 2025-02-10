//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PDGNumber.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe particle identifier.
 *
 * The Particle Data Group specifies a coding to uniquely identify
 * standard-model particle
 * types in "Monte Carlo Particle Numbering Scheme" section of \citep{pdg,
 * https://link.aps.org/doi/10.1103/PhysRevD.98.030001}.
 * These coded identifiers should generally not be treated like numbers: this
 * class prevents unintentional arithmetic and conversion.
 *
 * PDG numbers should only be used in host setup code (they should be converted
 * to ParticleId for use during runtime).
 */
class PDGNumber
{
  public:
    //! Construct with an invalid/unassigned value of zero
    constexpr PDGNumber() = default;

    //! Construct with the PDG value
    explicit constexpr PDGNumber(int val) : value_(val) {}

    //! True if value is nonzero
    explicit constexpr operator bool() const { return value_ != 0; }

    //! Get the PDG value
    constexpr int unchecked_get() const { return value_; }

    // Get the PDG value
    inline int get() const;

  private:
    int value_{0};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the PDG value.
 */
inline int PDGNumber::get() const
{
    CELER_ENSURE(*this);
    return value_;
}

//---------------------------------------------------------------------------//
// COMPARATORS
//---------------------------------------------------------------------------//
//! Test equality
inline constexpr bool operator==(PDGNumber lhs, PDGNumber rhs)
{
    return lhs.unchecked_get() == rhs.unchecked_get();
}

//! Test inequality
inline constexpr bool operator!=(PDGNumber lhs, PDGNumber rhs)
{
    return !(lhs == rhs);
}

//! Allow less-than comparison for sorting
inline constexpr bool operator<(PDGNumber lhs, PDGNumber rhs)
{
    return lhs.unchecked_get() < rhs.unchecked_get();
}

/*!
 * Unique standard model particle identifiers by the Particle Data Group.
 *
 * This namespace acts an enumeration for PDG codes that are used by the
 * various processes in Celeritas. (Unlike an enumeration, though, PDG codes
 * can be arbitary and aren't limited to the ones defined below.) They should
 * be extended as needed when new particle types are used by processes.
 *
 * PDG numbers between 81 and 100 are reserved for internal use.
 * The table shows which internal arbitrary numbers are currently defined:
 *
 * | Particle name | PDG |
 * | ------------- | --- |
 * | Generic ion   | 90  |
 */
namespace pdg
{
//---------------------------------------------------------------------------//
#define CELER_DEFINE_PDGNUMBER(NAME, VALUE) \
    inline constexpr PDGNumber NAME()       \
    {                                       \
        return PDGNumber{VALUE};            \
    }

//!@{
//! Particle Data Group Monte Carlo number codes.
// Sorted by (abs(val), val < 0)
CELER_DEFINE_PDGNUMBER(electron, 11)
CELER_DEFINE_PDGNUMBER(positron, -11)
CELER_DEFINE_PDGNUMBER(mu_minus, 13)
CELER_DEFINE_PDGNUMBER(mu_plus, -13)
CELER_DEFINE_PDGNUMBER(gamma, 22)
CELER_DEFINE_PDGNUMBER(ion, 90)
CELER_DEFINE_PDGNUMBER(pi_plus, 211)
CELER_DEFINE_PDGNUMBER(pi_minus, -211)
CELER_DEFINE_PDGNUMBER(kaon_plus, 321)
CELER_DEFINE_PDGNUMBER(kaon_minus, -321)
CELER_DEFINE_PDGNUMBER(proton, 2212)
CELER_DEFINE_PDGNUMBER(anti_proton, -2212)
CELER_DEFINE_PDGNUMBER(neutron, 2112)
CELER_DEFINE_PDGNUMBER(anti_neutron, -2112)
CELER_DEFINE_PDGNUMBER(he3, 1000020030)
CELER_DEFINE_PDGNUMBER(anti_he3, -1000020030)
CELER_DEFINE_PDGNUMBER(alpha, 1000020040)
CELER_DEFINE_PDGNUMBER(anti_alpha, -1000020040)
CELER_DEFINE_PDGNUMBER(deuteron, 1000010020)
CELER_DEFINE_PDGNUMBER(anti_deuteron, -1000010020)
CELER_DEFINE_PDGNUMBER(triton, 1000010030)
CELER_DEFINE_PDGNUMBER(anti_triton, -1000010030)
//!@}

#undef CELER_DEFINE_PDGNUMBER
//---------------------------------------------------------------------------//
}  // namespace pdg
}  // namespace celeritas

//---------------------------------------------------------------------------//
// STD::HASH SPECIALIZATION FOR HOST CODE
//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<>
struct hash<celeritas::PDGNumber>
{
    using argument_type = celeritas::PDGNumber;
    using result_type = std::size_t;
    result_type operator()(argument_type const& pdg) const noexcept
    {
        return std::hash<int>()(pdg.unchecked_get());
    }
};
}  // namespace std
//! \endcond
