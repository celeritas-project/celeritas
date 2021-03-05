//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Pie.hh"
#include "physics/base/Units.hh"
#include "Types.hh"
#include "physics/base/Types.hh"
#include "physics/material/Types.hh"

namespace celeritas
{
struct SingleCutoff
{
    units::MevEnergy energy{};
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared cutoff data.
 * 
 * The data is a vector of cutoff energy values for every particle type and 
 * material. 
 * 
 * \sa CutoffView
 * \sa CutoffParams
 */
template<Ownership W, MemSpace M>
struct CutoffParamsData
{
    template<class T>
    using Data = Pie<T, W, M>;

    // Backend storage
    Data<SingleCutoff> cutoffs; // [num_particles][num_materials]

    ParticleId::value_type num_particles{};
    MaterialId::value_type num_materials{};

    //// MEMBER FUNCTIONS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return cutoffs.size() == num_particles * num_materials
               && !cutoffs.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CutoffParamsData& operator=(const CutoffParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        this->cutoffs       = other.cutoffs;
        this->num_particles = other.num_particles;
        this->num_materials = other.num_materials;

        return *this;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
