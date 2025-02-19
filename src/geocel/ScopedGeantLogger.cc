//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantLogger.cc
//---------------------------------------------------------------------------//
#include "ScopedGeantLogger.hh"

#include <cctype>
#include <memory>
#include <mutex>
#include <regex>
#include <G4String.hh>
#include <G4Threading.hh>
#include <G4Types.hh>
#include <G4UImanager.hh>
#include <G4Version.hh>
#include <G4coutDestination.hh>

#include "corecel/Macros.hh"
#if G4VERSION_NUMBER >= 1120
#    include <G4ios.hh>
#    define CELER_G4SSBUF 0
#else
#    include <G4strstreambuf.hh>
#    define CELER_G4SSBUF 1
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
//! Get a string view matched by a regular expression
template<class T>
std::string_view to_string_view(std::sub_match<T> const& cs)
{
    if (!cs.matched)
    {
        return {};
    }
    return {&(*cs.first), static_cast<std::size_t>(cs.length())};
}

//---------------------------------------------------------------------------//
/*!
 * Log the actual message.
 */
G4int log_impl(G4String const& str, LogLevel level)
{
    static std::regex const err_warn_regex{
        R"regex(^\W*(\w+)?\s*(warning|error|!+|\*+)\W+)regex",
        std::regex::icase};

    static std::regex const info_regex{R"regex(^(\w+):\s+)regex"};

    std::smatch m;
    std::string_view msg;
    std::string_view source{"Geant4"};
    if (std::regex_search(str, m, err_warn_regex))
    {
        CELER_ASSERT(m.size() == 3);
        if (m[1].matched)
        {
            // Warning is coming from somewhere in particular
            source = to_string_view(m[1]);
        }

        // Strip the beginning text from the err/warning
        msg = to_string_view(m.suffix());
        // Update the warning level
        auto first_char = std::tolower(static_cast<unsigned char>(*m[2].first));
        if (first_char == 'w' || first_char == '*')
        {
            level = LogLevel::warning;
        }
        else if (first_char == 'e' || first_char == '!')
        {
            level = LogLevel::error;
        }
        else
        {
            CELER_ASSERT_UNREACHABLE();
        }
    }
    else if (std::regex_search(str, m, info_regex))
    {
        CELER_ASSERT(m.size() == 2);
        source = to_string_view(m[1]);
        msg = to_string_view(m.suffix());
    }
    else
    {
        msg = str;
    }

    // Output with dummy file/line
    ::celeritas::world_logger()({source, 0}, level) << trim(msg);

    // 0 for success, -1 for failure
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Send Geant4 log messages to Celeritas' world logger.
 */
class GeantLoggerAdapter final : public G4coutDestination
{
  public:
    // Assign to Geant handlers on construction
    GeantLoggerAdapter();
    CELER_DEFAULT_COPY_MOVE(GeantLoggerAdapter);
    ~GeantLoggerAdapter() final;

    // Handle error messages
    G4int ReceiveG4cout(G4String const& str) final;
    G4int ReceiveG4cerr(G4String const& str) final;

  private:
#if CELER_G4SSBUF
    G4coutDestination* saved_cout_{nullptr};
    G4coutDestination* saved_cerr_{nullptr};
#endif
};

//---------------------------------------------------------------------------//
/*!
 * Redirect geant4's stdout/cerr on construction.
 *
 * Note that all these buffers, and the UI pointers, are thread-local.
 */
GeantLoggerAdapter::GeantLoggerAdapter()
{
    // Make sure UI pointer has been instantiated, since its constructor
    // resets the cout destination
    if (!G4UImanager::GetUIpointer())
    {
        // Always-on debug assertion (not a "runtime" error but a
        // subtle programming logic error that always causes a crash)
        CELER_DEBUG_FAIL(
            R"(Geant4 logging cannot be changed after G4UImanager has been destroyed)",
            precondition);
    }

#if CELER_G4SSBUF
    saved_cout_ = G4coutbuf.GetDestination();
    saved_cerr_ = G4cerrbuf.GetDestination();
    G4coutbuf.SetDestination(this);
    G4cerrbuf.SetDestination(this);
#else
    // See Geant4 change global-V11-01-01
    G4iosSetDestination(this);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Restore iostream buffers on destruction.
 */
GeantLoggerAdapter::~GeantLoggerAdapter()
{
#if CELER_G4SSBUF
    G4coutbuf.SetDestination(saved_cout_);
    G4cerrbuf.SetDestination(saved_cerr_);
#else
    G4iosSetDestination(nullptr);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Process stdout message.
 */
G4int GeantLoggerAdapter::ReceiveG4cout(G4String const& str)
{
    return log_impl(str, LogLevel::diagnostic);
}

//---------------------------------------------------------------------------//
/*!
 * Process stderr message.
 */
G4int GeantLoggerAdapter::ReceiveG4cerr(G4String const& str)
{
    return log_impl(str, LogLevel::info);
}

//---------------------------------------------------------------------------//
//! Thread-local flag for "ownership" of the Geant4 logger
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
G4ThreadLocal bool g_adapter_active_{false};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas Geant4 logger.
 *
 * A global flag allows multiple logger adapters to be nested without
 * consequence.
 */
ScopedGeantLogger::ScopedGeantLogger()
{
    if (!g_adapter_active_)
    {
        g_adapter_active_ = true;
        logger_ = std::make_unique<GeantLoggerAdapter>();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous exception handler.
 */
ScopedGeantLogger::~ScopedGeantLogger()
{
    if (logger_)
    {
        logger_.reset();
        g_adapter_active_ = false;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
