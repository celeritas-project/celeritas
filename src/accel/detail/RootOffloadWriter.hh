//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/RootOffloadWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>

#include "corecel/Macros.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/phys/Primary.hh"

namespace celeritas
{
class ParticleParams;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Export primary data to ROOT.
 *
 * One TTree entry represents one primary.
 */
class RootOffloadWriter
{
  public:
    //!@{
    //! \name Type aliases
    using Primaries = std::vector<Primary>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

    // Construct with ROOT output filename
    explicit RootOffloadWriter(std::string const& root_output_name,
                               SPConstParticles params);

    // Export primaries to ROOT
    void operator()(Primaries const& primaries);

  private:
    //// DATA ////

    // Basic data types stored to ROOT to avoid the need of a dictionary
    struct RootOffloadPrimary
    {
        std::size_t event_id;
        std::size_t track_id;
        int particle;
        double energy;
        double time;
        std::array<double, 3> pos;
        std::array<double, 3> dir;
    };

    RootFileManager tfile_mgr_;
    SPConstParticles params_;
    UPRootAutoSave<TTree> ttree_;
    RootOffloadPrimary primary_;  // Temporary object stored to the ROOT TTree
    std::mutex write_mutex_;

    //// HELPER FUNCTIONS ////

    // Hardcoded TTree name and title
    char const* tree_name() { return "primaries"; }
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootOffloadWriter(std::string const&, SPConstParticles params)
{
    (void)sizeof(tfile_mgr_);
    (void)sizeof(params_);
    (void)sizeof(ttree_);
    (void)sizeof(primary_);
    (void)sizeof(write_mutex_);
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootOffloadWriter::operator()(Primaries const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
