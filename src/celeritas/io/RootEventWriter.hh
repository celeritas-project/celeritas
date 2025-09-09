//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootEventWriter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <array>

#include "corecel/Macros.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "celeritas/phys/Primary.hh"

#include "EventIOInterface.hh"

namespace celeritas
{
class ParticleParams;

//---------------------------------------------------------------------------//
/*!
 * Export primary data to ROOT.
 *
 * One TTree entry represents one primary.
 */
class RootEventWriter : public EventWriterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPRootFileManager = std::shared_ptr<RootFileManager>;
    //!@}

    // Construct with ROOT output filename
    RootEventWriter(SPRootFileManager root_file_manager,
                    SPConstParticles params);

    //! Prevent copying and moving
    CELER_DELETE_COPY_MOVE(RootEventWriter);
    ~RootEventWriter() override = default;

    // Export primaries to ROOT
    void operator()(VecPrimary const& primaries) override;

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

    SPRootFileManager tfile_mgr_;
    SPConstParticles params_;
    size_type event_id_;  // Contiguous event id
    UPRootTreeWritable ttree_;
    RootOffloadPrimary primary_;  // Temporary object stored to the ROOT TTree
    bool warned_mismatched_events_{false};

    //// HELPER FUNCTIONS ////

    // Hardcoded TTree name and title
    char const* tree_name() { return "primaries"; }
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootEventWriter::RootEventWriter(SPRootFileManager, SPConstParticles)
{
    CELER_DISCARD(tfile_mgr_);
    CELER_DISCARD(params_);
    CELER_DISCARD(event_id_);
    CELER_DISCARD(ttree_);
    CELER_DISCARD(primary_);
    CELER_DISCARD(warned_mismatched_events_);
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootEventWriter::operator()(VecPrimary const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
