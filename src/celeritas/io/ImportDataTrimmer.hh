//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportDataTrimmer.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/math/NumericLimits.hh"

#include "ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reduce the amount of imported/exported data for testing.
 */
class ImportDataTrimmer
{
  public:
    struct Input
    {
        //! Remove materials/elements which might affect the problem
        bool materials{false};
        //! Reduce the number of physics models and processes
        bool physics{false};
        //! Reduce the MuPPET table fidelity
        bool mupp{false};
        //! Maximum number of elements in a vector (approximate)
        std::size_t max_size{numeric_limits<std::size_t>::max()};
    };

  public:
    // Construct with a unit system
    explicit ImportDataTrimmer(Input const& inp);

    //!@{
    //! Trim imported data
    void operator()(ImportData* data);
    void operator()(ImportElement* data);
    void operator()(ImportGeoMaterial* data);
    void operator()(ImportModel* data);
    void operator()(ImportModelMaterial* data);
    void operator()(ImportMscModel* data);
    void operator()(ImportMuPairProductionTable* data);
    void operator()(ImportOpticalMaterial* data);
    void operator()(ImportOpticalModel* data);
    void operator()(ImportParticle* data);
    void operator()(ImportPhysMaterial* data);
    void operator()(ImportProcess* data);
    //!@}

    //!@{
    //! Trim objects
    void operator()(ImportPhysicsVector* data);
    void operator()(ImportPhysicsTable* data);
    void operator()(ImportPhysics2DVector* data);
    //!@}

  private:
    //// TYPES ////

    struct GridFilterer;

    //// DATA ////

    Input options_;

    //// HELPERS ////

    GridFilterer make_filterer(std::size_t vec_size) const;

    template<class T>
    void operator()(std::vector<T>* vec);

    template<class K, class T, class C, class A>
    void operator()(std::map<K, T, C, A>* m);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
