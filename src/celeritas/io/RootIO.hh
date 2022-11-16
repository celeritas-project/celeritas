//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/RootIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "celeritas/ext/detail/TRootUniquePtr.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/user/StepData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store step data into a ROOT file. Step data is colected using
 * \c operator() , which fills the step TTree. At any given time, the user may
 * call \c write() to flush existing in-memory filled data to disk. At
 * destruction time the TFile is closed, unless manually closed before via the
 * \c close() function call.
 *
 * \note This class is (currently) not meant to be used with multithread.
 */
class RootIO
{
  public:
    //!@{
    //! \name type aliases
    using SPParticleParams = std::shared_ptr<ParticleParams>;
    //!@}

    //// ROOT DATA ////
    struct TStepPoint
    {
        int    volume_id;
        double dir[3];
        double pos[3]; //!< [cm]
        double energy; //!< [MeV]
        double time;   //!< [s]
    };

    struct TStepData
    {
        int        event_id;
        int        track_id;
        int        action_id;
        int        pdg;
        int        track_step_count;
        TStepPoint points[2];
        double     energy_deposition; //!< [MeV]
        double     length;            //!< [cm]
    };

    //// METHODS ////

  public:
    // Construct with ROOT output filename
    explicit RootIO(const char* filename, SPParticleParams particles);

    // Close TFile at destruction
    ~RootIO();

    // Store step data in the TTree
    void store(HostCRef<StepStateData> steps);

    // Set number of entries stored in memory before being flushed to disk
    void set_auto_flush(long num_entries);

    // Get tfile_ to allow storing extra data (such as input information)
    TFile* tfile_get();

    // Close root file (can automatically be done at destruction)
    void close();

  private:
    SPParticleParams particles_;

    detail::TRootUniquePtr<TFile>   tfile_;
    detail::TRootUniquePtr<TTree>   step_tree_;
    detail::TRootUniquePtr<TBranch> step_branch_;
    TStepData                       tstep_;
};

//---------------------------------------------------------------------------//
#if !CELERITAS_USE_ROOT
inline RootIO::RootIO(const char* filename)
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootIO::~RootIO()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootIO::operator()()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootIO::write()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootIO::tfile_get()
{
    CELER_NOT_CONFIGURED("ROOT");
}

inline RootIO::close()
{
    CELER_NOT_CONFIGURED("ROOT");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
