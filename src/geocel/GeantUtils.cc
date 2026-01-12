//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantUtils.cc
//! \todo Move anything that's not geometry/materials to the celeritas lib
//---------------------------------------------------------------------------//
#include "GeantUtils.hh"

#include <G4ParticleDefinition.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER < 1070
#    include <G4MTRunManager.hh>
#endif

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the number of threads in a version-portable way.
 *
 * G4RunManager::GetNumberOfThreads isn't virtual before Geant4 v10.7.0 so we
 * need to explicitly dynamic cast to G4MTRunManager to get the number of
 * threads.
 *
 * In tasking mode, the result may be zero!
 */
int get_geant_num_threads(G4RunManager const& runman)
{
    // Default is 1 if not multithreaded
    int result{1};
#if G4VERSION_NUMBER >= 1070
    result = runman.GetNumberOfThreads();
#else
    if (auto const* runman_mt = dynamic_cast<G4MTRunManager const*>(&runman))
    {
        result = runman_mt->GetNumberOfThreads();
    }
#endif
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of threads from the global run manager.
 */
int get_geant_num_threads()
{
    auto* run_man = G4RunManager::GetRunManager();
    CELER_VALIDATE(run_man,
                   << "cannot query global thread count before G4RunManager "
                      "is created");
    return get_geant_num_threads(*run_man);
}

//---------------------------------------------------------------------------//
/*!
 * Get the Geant4 thread ID.
 */
int get_geant_thread_id()
{
    // Thread ID is -1 when running serially
    if (G4Threading::IsMultithreadedApplication())
    {
        return G4Threading::G4GetThreadId();
    }
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Validate the thread ID and threading model.
 */
void validate_geant_threading(size_type num_streams)
{
    auto thread_id = get_geant_thread_id();
    CELER_VALIDATE(thread_id >= 0,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is invalid (perhaps local offload is being built "
                      "on a non-worker thread?)");
    CELER_VALIDATE(static_cast<size_type>(thread_id) < num_streams,
                   << "Geant4 ThreadID (" << thread_id
                   << ") is out of range for the reported number of worker "
                      "threads ("
                   << num_streams << ")");

    // Check that OpenMP and Geant4 threading models don't collide
    if (CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK && !celeritas::device()
        && G4Threading::IsMultithreadedApplication())
    {
        auto msg = CELER_LOG(warning);
        msg << "Using multithreaded Geant4 with Celeritas track-level OpenMP "
               "parallelism";
        if (std::string const& nt_str = celeritas::getenv("OMP_NUM_THREADS");
            !nt_str.empty())
        {
            msg << "(OMP_NUM_THREADS=" << nt_str
                << "): CPU threads may be oversubscribed";
        }
        else
        {
            msg << ": forcing 1 Celeritas thread to Geant4 thread";
#ifdef _OPENMP
            omp_set_num_threads(1);
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Print a particle definition name and PDG.
 */
std::ostream& operator<<(std::ostream& os, StreamablePD const& ppd)
{
    if (ppd.pd)
    {
        os << '"' << ppd.pd->GetParticleName() << "\"@"
           << static_cast<void const*>(ppd.pd)
           << " (PDG = " << ppd.pd->GetPDGEncoding() << ')';
    }
    else
    {
        os << "{null G4ParticleDefinition}";
    }
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
