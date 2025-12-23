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
        return units::MevEnergy{1e+8};
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

    // Microscopic cross sections using G4PARTICLEXS/gamma nuclear (IAEA) data
    ElementItems<NonuniformGridRecord> xs_iaea;
    Items<real_type> reals;

    // Microscopic cross sections using parameterized CHIPS data at high energy
    ElementItems<NonuniformGridRecord> xs_chips;

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return scalars && !xs_iaea.empty() && !reals.empty()
               && xs_chips.size() == xs_iaea.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GammaNuclearData& operator=(GammaNuclearData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        xs_iaea = other.xs_iaea;
        xs_chips = other.xs_chips;
        reals = other.reals;

        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
