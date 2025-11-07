//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedGeantExceptionHandler.hh"

#include <regex>
#include <G4ExceptionSeverity.hh>
#include <G4RunManager.hh>
#include <G4StateManager.hh>
#include <G4Threading.hh>
#include <G4Types.hh>
#include <G4VExceptionHandler.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
int& local_eh_depth()
{
    static G4ThreadLocal int depth{0};
    return depth;
}

//---------------------------------------------------------------------------//
// NOTE: this is modified from test/corecel/StringSimplifier and could be
// turned into a helper class
std::string strip_ansi(std::string const& s)
{
    static std::regex re("(?:\033\\[[0-9;]*m|\n|:\\s+)");

    std::string result;
    result.reserve(s.size());
    auto current = s.cbegin();
    std::smatch match;

    // Find all matches and process them one by one
    while (std::regex_search(current, s.cend(), match, re))
    {
        // Add the text between the last match and this one
        result.append(current, match[0].first);
        if (*match[0].first == '\033')
        {
            // Omit ANSI
        }
        else
        {
            // Replace newline/separators
            result += " / ";
        }
        // Update current position
        current = match[0].second;
    }

    // Add remaining text after the last match
    result.append(current, s.cend());

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process Geant4 exceptions with Celeritas.
 *
 * The Geant exception handler base class changes global state in its
 * constructor (assigning "this") so this class instance must stay in scope
 * once created. There is no way to save or restore the previous handler.
 * Furthermore, creating a G4RunManagerKernel also resets the exception
 * handler, so errors thrown during setup *CANNOT* be caught by celeritas, and
 * this class can only be used after creating the G4RunManager.
 */
class GeantExceptionHandler final : public G4VExceptionHandler
{
  public:
    using Handler = ScopedGeantExceptionHandler::StdExceptionHandler;

    GeantExceptionHandler(Handler&& handle) : handle_{std::move(handle)} {}

    // Accept error codes from geant4
    G4bool Notify(char const* originOfException,
                  char const* exceptionCode,
                  G4ExceptionSeverity severity,
                  char const* description) final;

  private:
    Handler handle_;
};

//---------------------------------------------------------------------------//
/*!
 * Propagate exceptions to Celeritas.
 */
G4bool GeantExceptionHandler::Notify(char const* origin_of_exception,
                                     char const* exception_code,
                                     G4ExceptionSeverity severity,
                                     char const* description)
{
    CELER_EXPECT(origin_of_exception);
    CELER_EXPECT(exception_code);

    // Construct message
    auto err = RuntimeError{[&] {
        RuntimeErrorDetails details;
        details.which = "Geant4";
        details.what = description;
        details.condition = exception_code;
        details.file = origin_of_exception;
        return details;
    }()};

    switch (severity)
    {
        case FatalException:
        case FatalErrorInArgument:
        case RunMustBeAborted:
        case EventMustBeAborted:
            // Severe or initialization error
            CELER_LOG_LOCAL(debug)
                << "Handling exception: " << strip_ansi(err.what());
            CELER_TRY_HANDLE(throw err, handle_);
            if (auto* run_man = G4RunManager::GetRunManager())
            {
                CELER_LOG_LOCAL(critical) << "Aborting run due to exception ("
                                          << exception_code << ")";
                run_man->AbortRun();
            }
            break;
        case JustWarning: {
            // Display a message: log destination depends on whether we're
            // actually running particles and if the thread is a worker (or if
            // it's not multithreaded). Setup errors get sent to world; runtime
            // errors are sent to self.
            bool const is_runtime_error
                = (G4StateManager::GetStateManager()->GetCurrentState()
                   == G4State_EventProc)
                  && (G4Threading::IsWorkerThread()
                      || !G4Threading::IsMultithreadedApplication());
            auto& log = (is_runtime_error ? celeritas::self_logger()
                                          : celeritas::world_logger());
            log(CELER_CODE_PROVENANCE, ::celeritas::LogLevel::error)
                << err.what();
            break;
        }
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Return "true" to cause Geant4 to crash the program, or "false" to let it
    // know that we've handled the exception.
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas Geant4 exception handler.
 *
 * The base class of the exception handler calls SetExceptionHandler...
 */
ScopedGeantExceptionHandler::ScopedGeantExceptionHandler(
    StdExceptionHandler handle)
{
    CELER_LOG_LOCAL(debug) << "Creating scoped G4 exception handler (depth "
                           << local_eh_depth()++ << ")";
    // Get thread-local state manager, to which the handler assigns itself
    auto* state_mgr = G4StateManager::GetStateManager();
    CELER_ASSERT(state_mgr);
    previous_ = state_mgr->GetExceptionHandler();
    if (!handle)
    {
        handle = std::rethrow_exception;
    }
    current_ = std::make_unique<GeantExceptionHandler>(std::move(handle));
    CELER_ENSURE(state_mgr->GetExceptionHandler() == current_.get());
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous exception handler.
 */
ScopedGeantExceptionHandler::~ScopedGeantExceptionHandler()
{
    CELER_LOG_LOCAL(debug) << "Destroying scoped G4 exception handler (depth "
                           << --local_eh_depth() << ")";
    auto* state_mgr = G4StateManager::GetStateManager();
    if (state_mgr->GetExceptionHandler() == current_.get())
    {
        state_mgr->SetExceptionHandler(previous_);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
