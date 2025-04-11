//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportDataTrimmer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <vector>

#include "corecel/math/NumericLimits.hh"
#include "celeritas/inp/Grid.hh"

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
        //! Maximum number of elements in a vector (approximate)
        std::size_t max_size{numeric_limits<std::size_t>::max()};

        //! Remove materials/elements which might affect the problem
        bool materials{false};
        //! Reduce the number of physics models and processes
        bool physics{false};
        //! Reduce the MuPPET table fidelity
        bool mupp{false};
    };

  public:
    // Construct with a unit system
    explicit ImportDataTrimmer(Input const& inp);

    //!@{
    //! Trim imported data
    void operator()(ImportData& data);
    void operator()(ImportElement& data);
    void operator()(ImportGeoMaterial& data);
    void operator()(ImportModel& data);
    void operator()(ImportModelMaterial& data);
    void operator()(ImportMscModel& data);
    void operator()(ImportLivermorePE& data);
    void operator()(ImportLivermoreSubshell& data);
    void operator()(ImportAtomicRelaxation& data);
    void operator()(ImportMuPairProductionTable& data);
    void operator()(ImportOpticalMaterial& data);
    void operator()(ImportOpticalModel& data);
    void operator()(ImportParticle& data);
    void operator()(ImportPhysMaterial& data);
    void operator()(ImportProcess& data);
    //!@}

    //!@{
    //! Trim objects
    void operator()(inp::Grid& data);
    void operator()(ImportPhysicsTable& data);
    void operator()(inp::TwodGrid& data);
    //!@}

  private:
    //// TYPES ////

    using size_type = std::size_t;
    struct GridFilterer;

    //// DATA ////

    Input options_;

    //// HELPERS ////

    GridFilterer make_filterer(size_type vec_size) const;

    template<class T>
    void operator()(std::vector<T>& vec);

    template<class K, class T, class C, class A>
    void operator()(std::map<K, T, C, A>& m);

    template<class T>
    void for_each(std::vector<T>& vec);

    template<class K, class T, class C, class A>
    void for_each(std::map<K, T, C, A>& m);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
