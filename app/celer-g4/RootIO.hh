//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RootIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <string>
#include <vector>
#include <G4ThreadLocalSingleton.hh>
#include <G4Types.hh>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/io/EventData.hh"

class TFile;
class TTree;
class TBranch;
class G4Event;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*
 * Example of writing data to ROOT output.
 */
class RootIO
{
    friend class G4ThreadLocalSingleton<RootIO>;

  public:
#if CELERITAS_USE_ROOT
    // Whether ROOT output is enabled
    static bool use_root();
#else
    // ROOT is never enabled if ROOT isn't available
    constexpr static bool use_root() { return false; }
#endif

    // Return non-owning pointer to a singleton
    static RootIO* Instance();

    // Write sensitive hits of a G4Event to ROOT output file
    void Write(G4Event const* event);

    // Add detector name to map of sensitive detectors
    void AddSensitiveDetector(std::string name);

    // Close or merge output files
    void Close();

  private:
    // Construct by initializing TFile and TTree on each worker thread
    RootIO();
    RootIO(RootIO&&) = default;

    // Assignment operator
    RootIO& operator=(RootIO&&) = default;

    // Default destructor
    ~RootIO() = default;

    //// HELPER FUNCTIONS ////

    // Fill and write an EventData object
    void WriteObject(EventData* hit_event);

    // Merge ROOT files from multiple worker threads
    void Merge();

    // Store a new TTree mapping detector ID and name
    void StoreSdMap(TFile* file);

    //! ROOT TTree split level
    static constexpr short int SplitLevel() { return 99; }

    //! ROOT TTree name
    static char const* TreeName() { return "events"; }

    //// DATA ////

    std::string file_name_;
    std::unique_ptr<TFile> file_;
    std::unique_ptr<TTree> tree_;
    TBranch* event_branch_{nullptr};

    // Map sensitive detectors to contiguous IDs
    // Used by celeritas/io/EventData.hh
    int detector_id_{-1};
    std::map<std::string, int> detector_name_id_map_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootIO* RootIO::Instance()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootIO::Write(G4Event const*)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootIO::AddSensitiveDetector(std::string)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline void RootIO::Close()
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
