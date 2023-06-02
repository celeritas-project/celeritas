//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"

#include "HeuristicGeoData.hh"

namespace celeritas
{
template<template<Ownership, MemSpace> class S, MemSpace M>
class CollectionStateStore;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Manage a "heuristic" stepper-like test that accumulates path length.
 */
class HeuristicGeoTestBase : public GlobalGeoTestBase, public OnlyGeoTestBase
{
  public:
    //!@{
    //! \name Type aliases
    template<MemSpace M>
    using StateStore = CollectionStateStore<HeuristicGeoStateData, M>;
    template<MemSpace M>
    using PathLengthRef
        = Collection<real_type, Ownership::reference, M, VolumeId>;
    using SpanConstReal = Span<real_type const>;
    using SpanConstStr = Span<const std::string>;
    //!@}

    //// INTERFACE ////

    //! Construct problem-specific attributes (sampling box etc)
    virtual HeuristicGeoScalars build_scalars() const = 0;
    //! Get the number of steps to execute
    virtual size_type num_steps() const = 0;
    //! Build a list of volumes to compare average paths
    virtual SpanConstStr reference_volumes() const;
    //! Return the vector of path lengths mapped by sorted volume name
    virtual SpanConstReal reference_avg_path() const;

  protected:
    //// TEST EXECUTION ////

    //!@{
    //! Run tracks on device or host and compare the resulting path length
    void run_host(size_type num_states, real_type tolerance);
    void run_device(size_type num_states, real_type tolerance);
    //!@}

  private:
    //// DATA ////

    // Backend data for default reference_volumes implementation
    mutable std::vector<std::string> temp_str_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    HeuristicGeoParamsData<Ownership::const_reference, M> build_test_params();

    template<MemSpace M>
    std::vector<real_type>
    get_avg_path(PathLengthRef<M> path, size_type num_states) const;

    std::vector<real_type> get_avg_path_impl(std::vector<real_type> const& path,
                                             size_type num_states) const;
};

//---------------------------------------------------------------------------//
//! Run on device
void heuristic_test_execute(DeviceCRef<HeuristicGeoParamsData> const&,
                            DeviceRef<HeuristicGeoStateData> const&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
