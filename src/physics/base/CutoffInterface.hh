//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Collection.hh"
#include "physics/base/Units.hh"
#include "physics/base/Types.hh"
#include "physics/material/Types.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store secondary cutoff information.
 */
struct ParticleCutoff
{
    units::MevEnergy energy{}; //!< Converted range value
    real_type        range{};  //!< [cm]
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
    using Items = Collection<T, W, M>;

    // Backend storage
    Items<ParticleCutoff> cutoffs; // [num_materials][num_particles]

    ParticleId::size_type num_particles;
    MaterialId::size_type num_materials;

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
