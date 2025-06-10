//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ExceptionConverter.cc
//---------------------------------------------------------------------------//
#include "ExceptionConverter.hh"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <G4Exception.hh>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueIdIO.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/QuantityIO.hh"
#include "corecel/sys/Environment.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "SharedParams.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool determine_strip()
{
    if (!celeritas::getenv("CELER_STRIP_SOURCEDIR").empty())
    {
        return true;
    }
    return static_cast<bool>(CELERITAS_DEBUG);
}

//---------------------------------------------------------------------------//
//! Try removing up to and including the filename from the reported path.
std::string strip_source_dir(std::string const& filename)
{
    static bool const do_strip = determine_strip();
    if (!do_strip)
    {
        // Don't strip in debug mode
        return filename;
    }

    std::string::size_type max_pos = 0;
    for (std::string const path : {"src/", "app/", "test/"})
    {
        auto pos = filename.rfind(path);

        if (pos != std::string::npos)
        {
            pos += path.size() - 1;
            max_pos = std::max(max_pos, pos);
        }
    }
    if (max_pos == 0)
    {
        // No telling where the filename is from...
        return filename;
    }

    return filename.substr(max_pos + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Write a detailed state message.
 */
void log_state(Logger::Message& msg,
               KernelContextException const& kce,
               SharedParams const* params)
{
    auto core_params = [params]() -> CoreParams const* {
        if (!params)
            return nullptr;
        if (!*params)
            return nullptr;
        return params->Params().get();
    }();

    if (core_params && kce.particle())
    {
        auto const& par_params = *core_params->particle();
        msg << "\n- Particle type: " << par_params.id_to_label(kce.particle())
            << " (PDG=" << par_params.id_to_pdg(kce.particle()).get()
            << ",  ID=" << kce.particle() << ')';
    }
    else
    {
        msg << "\n- Particle type ID: " << kce.particle();
    }
    msg << "\n- Energy: " << kce.energy() << "\n- Position: " << kce.pos()
        << " [" << units::NativeTraits::Length::label() << "]"
        << "\n- Direction: " << kce.dir();

    if (core_params && kce.volume())
    {
        auto const& geo_params = *core_params->geometry();
        msg << "\n- Volume: " << geo_params.volumes().at(kce.volume())
            << " (ID=" << kce.volume() << ')';
    }
    else
    {
        msg << "\n- Volume ID: " << kce.volume();
    }

    if (kce.surface())
    {
        msg << "\n- Surface ID: " << kce.surface();
    }

    msg << "\n- Step counter: " << kce.num_steps();
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Capture the current exception and convert it to a G4Exception call.
 */
void ExceptionConverter::operator()(std::exception_ptr eptr) const
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (RichContextException const& e)
    {
        {
            auto msg = CELER_LOG_LOCAL(critical);
            msg << "The following error is from: " << e.what();
            if (auto* kce = dynamic_cast<KernelContextException const*>(&e))
            {
                // Try to write a detailed message with particle state
                try
                {
                    log_state(msg, *kce, params_);
                }
                catch (std::exception const& e)
                {
                    // Error writing the state
                    msg << "\n[error while exporting state: " << e.what()
                        << "]";
                }
                catch (...)  // NOLINT(bugprone-empty-catch)
                {
                    /* Do nothing */
                }
            }
        }
        try
        {
            std::rethrow_if_nested(e);
        }
        catch (...)
        {
            return (*this)(std::current_exception());
        }
    }
    catch (RuntimeError const& e)
    {
        // Translate a runtime error into a G4Exception call
        std::string const where = [&d = e.details()] {
            std::ostringstream os;
            if (!d.file.empty())
            {
                os << strip_source_dir(d.file);
            }
            if (d.line != 0)
            {
                os << ':' << d.line;
            }
            return std::move(os).str();
        }();

        std::string const what = [&d = e.details()] {
            std::ostringstream os;
            os << "Celeritas " << (d.which ? d.which : "unknown") << " error: ";
            if (cstring_equal(d.which, RuntimeError::not_config_err_str))
            {
                os << "required dependency is disabled in this build: ";
            }
            else if (cstring_equal(d.which, RuntimeError::not_impl_err_str))
            {
                os << "feature is not yet implemented: ";
            }
            os << d.what;

            return std::move(os).str();
        }();
        G4Exception(where.c_str(), err_code_, FatalException, what.c_str());
    }
    catch (DebugError const& e)
    {
        // Translate a *debug* error
        std::ostringstream where;
        where << strip_source_dir(e.details().file) << ':' << e.details().line;
        std::ostringstream what;
        what << to_cstring(e.details().which) << ": " << e.details().condition;
        G4Exception(
            where.str().c_str(), err_code_, FatalException, what.str().c_str());
    }
    catch (std::runtime_error const& e)
    {
        this->convert_device_exceptions(std::current_exception());
    }
    // (Any other errors will be rethrown and abort the program.)
}

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * No other exceptions are caught when device code is disabled.
 *
 * See ExceptionConverter.cu for the CUDA implementation of this.
 */
inline void
ExceptionConverter::convert_device_exceptions(std::exception_ptr eptr) const
{
    std::rethrow_exception(eptr);
}

#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas
