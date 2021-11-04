//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffData.hh
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
 * Secondary production cuts are stored for every material and for only the
 * particle types to which production cuts apply. Currently production cuts are
 * only needed for electrons and photons (protons are unused and positrons
 * cannot have a cutoff).
 *
 * \sa CutoffView
 * \sa CutoffParams
 */
template<Ownership W, MemSpace M>
struct CutoffParamsData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    // Backend storage
    Items<ParticleCutoff> cutoffs; //!< [num_materials][num_particles]

    // Direct address table for mapping particle ID to index in cutoffs
    ParticleItems<size_type> id_to_index;

    ParticleId::size_type num_particles; //!< Particles with production cuts
    MaterialId::size_type num_materials; //!< All materials in the problem

    //// MEMBER FUNCTIONS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return cutoffs.size() == num_particles * num_materials
               && !cutoffs.empty() && !id_to_index.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CutoffParamsData& operator=(const CutoffParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        this->cutoffs       = other.cutoffs;
        this->id_to_index   = other.id_to_index;
        this->num_particles = other.num_particles;
        this->num_materials = other.num_materials;

        return *this;
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas
