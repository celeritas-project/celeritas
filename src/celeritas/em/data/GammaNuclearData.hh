//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/GammaNuclearData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Scalar data for the gamma-nuclear model.
 */
struct GammaNuclearScalars
{
    // Particle IDs
    ParticleId gamma_id;

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{5e+4};
    }

    //! Maximum energy limit for PARTICLEXS/gamma cross sections [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_low_energy()
    {
        return units::MevEnergy{130};
    }

    //! Whether data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(gamma_id);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for calculating micro (element) cross sections.
 */
template<Ownership W, MemSpace M>
struct GammaNuclearData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    // Scalar data
    GammaNuclearScalars scalars;

    // Microscopic (element) cross section data (G4PARTICLEXS/gamma/inelZ)
    ElementItems<NonuniformGridRecord> micro_xs;

    // Backend data
    Items<real_type> reals;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !micro_xs.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GammaNuclearData& operator=(GammaNuclearData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        micro_xs = other.micro_xs;
        reals = other.reals;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
