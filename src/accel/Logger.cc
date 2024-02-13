//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/Logger.cc
//---------------------------------------------------------------------------//
#include "Logger.hh"

#include <algorithm>
#include <functional>
#include <mutex>
#include <string>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <G4Version.hh>
#include <G4ios.hh>

#if G4VERSION_NUMBER < 1070
#    include <celeritas/ext/GeantSetup.hh>
#endif
#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "geocel/GeantUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
class MtLogger
{
  public:
    explicit MtLogger(int num_threads);
    void operator()(LogProvenance prov, LogLevel lev, std::string msg);

  private:
    int num_threads_;
};

//---------------------------------------------------------------------------//
MtLogger::MtLogger(int num_threads) : num_threads_(num_threads)
{
    CELER_EXPECT(num_threads_ >= 0);
}

//---------------------------------------------------------------------------//
void MtLogger::operator()(LogProvenance prov, LogLevel lev, std::string msg)
{
    auto& cerr = G4cerr;

    {
        // Write the file name up to the last directory component
        auto last_slash = std::find(prov.file.rbegin(), prov.file.rend(), '/');
        if (!prov.file.empty() && last_slash == prov.file.rend())
        {
            --last_slash;
        }

        // Output problem line/file for debugging or high level
        cerr << color_code('x')
             << std::string(last_slash.base(), prov.file.end());
        if (prov.line)
            cerr << ':' << prov.line;
        cerr << color_code(' ') << ": ";
    }

    int local_thread = G4Threading::G4GetThreadId();
    if (local_thread >= 0)
    {
        // Logging from a worker thread
        if (CELER_UNLIKELY(local_thread >= num_threads_))
        {
            // In tasking or potentially other contexts, the max thread might
            // not be known. Update it here for better output.
            static std::mutex thread_update_mutex;
            std::lock_guard scoped_lock{thread_update_mutex};
            num_threads_ = std::max(local_thread + 1, num_threads_);
        }

        cerr << color_code('W') << '[' << local_thread + 1 << '/'
             << num_threads_ << "] " << color_code(' ');
    }
    cerr << to_color_code(lev) << to_cstring(lev) << ": " << color_code(' ')
         << msg << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct a logger that will redirect Celeritas messages through Geant4.
 *
 * This logger writes the current thread (and maximum number of threads) in
 * each output message, and sends each message through the thread-local \c
 * G4cerr.
 *
 * In the \c main of your application's exectuable, set the "process-local"
 * (MPI-aware) logger:
 * \code
    celeritas::self_logger() = celeritas::MakeMTLogger(*run_manager);
   \endcode
 */
Logger MakeMTLogger(G4RunManager const& runman)
{
    Logger log(MpiCommunicator{}, MtLogger{get_geant_num_threads(runman)});

    log.level(log_level_from_env("CELER_LOG_LOCAL"));
    return log;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
