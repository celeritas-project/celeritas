//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LoggerImpl.cc
//---------------------------------------------------------------------------//
#include "LoggerImpl.hh"

#include <G4Threading.hh>
#include <G4ios.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/LoggerTypes.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Try removing up to and including the filename from the reported path.
std::ostream& operator<<(std::ostream& os, CleanedProvenance ssd)
{
    static bool const do_strip = [] {
        auto result = getenv_flag("CELER_STRIP_SOURCEDIR", !CELERITAS_DEBUG);
        return result.value;
    }();

    std::string_view::size_type max_pos = 0;
    if (do_strip)
    {
        for (std::string_view const path : {"src/", "app/", "test/"})
        {
            auto pos = ssd.filename.rfind(path);

            if (pos != std::string_view::npos)
            {
                pos += path.size() - 1;
                max_pos = std::max(max_pos, pos);
            }
        }
    }

    if (max_pos != 0)
    {
        // Only print after the last found path component
        os << ssd.filename.substr(max_pos + 1);
    }
    else
    {
        // Print entire filename
        os << ssd.filename;
    }

    if (ssd.line > 0)
    {
        os << ':' << ssd.line;
    }

    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Write a color-annotated message.
 */
std::ostream& operator<<(std::ostream& os, ColorfulLogMessage clm)
{
    os << to_color_code(clm.lev) << to_cstring(clm.lev);
    if (!clm.prov.file.empty())
    {
        os << color_code('x') << "@"
           << CleanedProvenance{clm.prov.file, clm.prov.line};
    }
    os << color_code(' ') << ": " << clm.msg;
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with number of threads.
 */
MtSelfWriter::MtSelfWriter(int num_threads) : num_threads_(num_threads)
{
    CELER_EXPECT(num_threads_ >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Write output with thread ID.
 */
void MtSelfWriter::operator()(LogProvenance prov,
                              LogLevel lev,
                              std::string msg) const
{
    auto& cerr = G4cerr;

    int local_thread = G4Threading::G4GetThreadId();
    if (local_thread >= 0)
    {
        // Logging from a worker thread
        cerr << color_code('W') << '[';
        if (CELER_UNLIKELY(local_thread >= num_threads_))
        {
            // Possibly running with tasking
            cerr << local_thread + 1;
        }
        else
        {
            cerr << local_thread + 1 << '/' << num_threads_;
        }
        cerr << "] ";
    }
    else
    {
        // Logging "local" message from the master thread!
        cerr << color_code('W') << "[M!] ";
    }
    cerr << ColorfulLogMessage{prov, lev, msg} << std::endl;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
