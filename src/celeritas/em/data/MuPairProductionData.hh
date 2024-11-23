//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/MuPairProductionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/grid/TwodGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * IDs used by muon pair production.
 */
struct MuPairProductionIds
{
    ParticleId mu_minus;
    ParticleId mu_plus;
    ParticleId electron;
    ParticleId positron;

    //! Whether the IDs are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return mu_minus && mu_plus && electron && positron;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Sampling table for electron-positron pair production by muons.
 *
 * The value grids are organized by atomic number, where the Z grid is a
 * hardcoded, problem-independent set of atomic numbers equally spaced in \f$
 * \log Z \f$ that is linearly interpolated on. Each 2D grid is:
 * - x: logarithm of the energy [MeV] of the incident charged particle
 * - y: logarithm of the ratio of the energy transfer to the incident particle
 *   energy
 * - value: CDF calculated from the differential cross section
 */
template<Ownership W, MemSpace M>
struct MuPairProductionTableData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// MEMBER DATA ////

    ItemRange<real_type> logz_grid;
    Items<TwodGridData> grids;

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !logz_grid.empty()
               && logz_grid.size() == grids.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    MuPairProductionTableData&
    operator=(MuPairProductionTableData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        logz_grid = other.logz_grid;
        grids = other.grids;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Constant data for the muon pair production interactor.
 */
template<Ownership W, MemSpace M>
struct MuPairProductionData
{
    //// MEMBER DATA ////

    //! Particle IDs
    MuPairProductionIds ids;

    //! Electron mass [MeV / c^2]
    units::MevMass electron_mass;

    // Sampling table storage
    MuPairProductionTableData<W, M> table;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return ids && electron_mass > zero_quantity() && table;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    MuPairProductionData& operator=(MuPairProductionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        ids = other.ids;
        electron_mass = other.electron_mass;
        table = other.table;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
