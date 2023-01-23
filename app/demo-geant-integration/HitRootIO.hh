//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/HitRootIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4ThreadLocalSingleton.hh>
#include <G4Types.hh>
#include <G4VHit.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

class TFile;
class TTree;
class TBranch;
class G4Event;

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Example of output event structure stored in ROOT.
 */
class HitRootEvent
{
  public:
    //!@{
    //! \name Type aliases
    using HitContainer = std::map<std::string, std::vector<G4VHit*>>;
    //!@}

  public:
    HitRootEvent() = default;

    void SetEventID(int id) { event_id_ = id; }
    HitContainer* GetHCMap() { return &hcmap_; }

  private:
    int event_id_{0};
    HitContainer hcmap_;
};

//---------------------------------------------------------------------------//
/*
 * Example of writing sensitive hits to ROOT output
 */
class HitRootIO
{
    friend class G4ThreadLocalSingleton<HitRootIO>;

  public:
    static HitRootIO* GetInstance();

    // Write sensitive hits of a G4Event to ROOT output file
    void WriteHits(G4Event const* event);

    // Close or merge output files
    void Close();

  private:
    // Set up output data and file
    HitRootIO();

    //// HELPER FUNCTIONS ////

    // Fill and write a HitRootEvent object
    void WriteObject(HitRootEvent* hit_event);

    // Merge ROOT files from multiple threads
    void Merge();

    //! The split level of ROOT TTree
    static constexpr short int SplitLevel() { return 99; }

  private:
    bool init_branch_{false};
    std::string file_name_;
    TFile* file_{nullptr};
    TTree* tree_{nullptr};
    TBranch* event_branch_{nullptr};
};

#if !CELERITAS_USE_ROOT
inline HitRootIO* HitRootIO::GetInstance()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void HitRootIO::WriteHits(G4Event const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void HitRootIO::Close()
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace demo_geant
