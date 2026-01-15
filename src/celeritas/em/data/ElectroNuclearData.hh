//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/ElectroNuclearData.hh
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
 * Scalar data for the electro-nuclear model.
 */
struct ElectroNuclearScalars
{
    // Particle IDs
    ParticleId electron_id;
    ParticleId positron_id;

    //! Model's minimum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_valid_energy()
    {
        return units::MevEnergy{1e+2};
    }

    //! Model's maximum energy limit [MeV]
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_valid_energy()
    {
        return units::MevEnergy{1e+8};
    }

    //! Whether data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return electron_id && positron_id;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for calculating micro (element) cross sections.
 */
template<Ownership W, MemSpace M>
struct ElectroNuclearData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ElementItems = Collection<T, W, M, ElementId>;

    //// MEMBER DATA ////

    // Scalar data
    ElectroNuclearScalars scalars;

    // Microscopic cross sections using parameterized data
    ElementItems<NonuniformGridRecord> micro_xs;
    Items<real_type> reals;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !micro_xs.empty() && !reals.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    ElectroNuclearData& operator=(ElectroNuclearData<W2, M2> const& other)
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
