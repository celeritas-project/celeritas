//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/data/NeutronCaptureData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data for calculating micro (element) cross sections.
 */
template<Ownership W, MemSpace M>
struct NeutronCaptureData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    //! ID of a neutron
    ParticleId neutron_id;

    //! Microscopic (element) cross section data (G4PARTICLEXS/neutron/capZ)
    ElementItems<NonuniformGridRecord> micro_xs;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{20};
    }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return neutron_id && !micro_xs.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    NeutronCaptureData& operator=(NeutronCaptureData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        neutron_id = other.neutron_id;
        micro_xs = other.micro_xs;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
