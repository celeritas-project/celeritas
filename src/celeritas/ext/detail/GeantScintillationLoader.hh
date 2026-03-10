//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantScintillationLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <string>

#include "corecel/inp/Distributions.hh"
#include "celeritas/inp/OpticalPhysics.hh"

namespace celeritas
{
namespace detail
{
class GeantMaterialPropertyGetter;
class GeantOpticalMatHelper;
//---------------------------------------------------------------------------//
/*!
 * Populate \c inp::ScintillationProcess data for one optical material.
 *
 * Construct with a reference to the process data to be filled. Call
 * \c operator() for each optical material.
 */
class GeantScintillationLoader
{
  public:
    //! Construct with scintillation process to populate
    explicit GeantScintillationLoader(inp::ScintillationProcess& process);

    //! Load scintillation data for one optical material
    void operator()(GeantOpticalMatHelper const& helper);

  private:
    inp::ScintillationProcess& process_;

    //// HELPER FUNCTIONS ////

    // Load scintillation data for one optical material (implementation)
    void load_one(GeantOpticalMatHelper const& helper);

    static std::optional<inp::NormalDistribution>
    load_gaussian(GeantMaterialPropertyGetter const& get,
                  std::string const& prefix,
                  std::string const& suffix);

    static std::optional<inp::ScintillationSpectrum>
    load_spectrum(GeantMaterialPropertyGetter const& get,
                  std::string const& prefix,
                  std::string const& suffix);
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
