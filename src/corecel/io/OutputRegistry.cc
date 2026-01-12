//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputRegistry.cc
//---------------------------------------------------------------------------//
#include "OutputRegistry.hh"

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <nlohmann/json.hpp>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/BuildOutput.hh"
#include "corecel/io/EnumStringMapper.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/DeviceIO.json.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/EnvironmentIO.json.hh"
#include "corecel/sys/KernelRegistry.hh"
#include "corecel/sys/KernelRegistryIO.json.hh"
#include "corecel/sys/MemRegistry.hh"
#include "corecel/sys/MemRegistryIO.json.hh"

#include "JsonPimpl.hh"
#include "Logger.hh"  // IWYU pragma: keep
#include "OutputInterface.hh"
#include "OutputInterfaceAdapter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Add an interface for writing.
 */
void OutputRegistry::insert(SPConstInterface interface)
{
    CELER_EXPECT(interface);
    CELER_EXPECT(interface->category() != Category::size_);

    auto label = interface->label();
    CELER_VALIDATE(!label.empty(), << "empty label for output interface");

    Category cat = interface->category();
    auto [prev, inserted]
        = interfaces_[cat].insert({std::string{label}, std::move(interface)});
    CELER_VALIDATE(inserted,
                   << "duplicate output entry '" << prev->first
                   << "' for category '" << cat << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Output all classes to a JSON object.
 */
void OutputRegistry::output(JsonPimpl* j) const
{
    nlohmann::json result;

    for (auto cat : range(Category::size_))
    {
        nlohmann::json cat_result;
        for (auto const& kv : interfaces_[cat])
        {
            // Hack for inlining input/result outputs in Transporter: to be
            // removed when the individual interfaces are converted to
            // OutputInterface
            bool is_global = (kv.first == "*");
            JsonPimpl json_wrap;
            if (is_global)
            {
                json_wrap.obj = std::move(cat_result);
                kv.second->output(&json_wrap);
                cat_result = std::move(json_wrap.obj);
            }
            else
            {
                kv.second->output(&json_wrap);
                cat_result[kv.first] = std::move(json_wrap.obj);
            }
        }
        if (!cat_result.empty())
        {
            // Add category to the final output
            result[to_cstring(cat)] = std::move(cat_result);
        }
    }

    j->obj = std::move(result);
}

//---------------------------------------------------------------------------//
/*!
 * Output all classes to a JSON object that's written to the given stream.
 */
void OutputRegistry::output(std::ostream* os) const
{
    JsonPimpl json_wrap;
    this->output(&json_wrap);
    *os << json_wrap.obj.dump();
}

//---------------------------------------------------------------------------//
/*!
 * Whether no output has been registered.
 */
bool OutputRegistry::empty() const
{
    return std::all_of(interfaces_.begin(),
                       interfaces_.end(),
                       [](auto const& m) { return m.empty(); });
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Add interfaces for writing system diagnostics.
 */
void insert_system_diagnostics(OutputRegistry& output_reg)
{
    output_reg.insert(OutputInterfaceAdapter<Device>::from_const_ref(
        OutputInterface::Category::system, "device", celeritas::device()));
    output_reg.insert(OutputInterfaceAdapter<KernelRegistry>::from_const_ref(
        OutputInterface::Category::system,
        "kernels",
        celeritas::kernel_registry()));
    output_reg.insert(OutputInterfaceAdapter<MemRegistry>::from_const_ref(
        OutputInterface::Category::system, "memory", celeritas::mem_registry()));
    output_reg.insert(OutputInterfaceAdapter<Environment>::from_const_ref(
        OutputInterface::Category::system, "environ", celeritas::environment()));
    output_reg.insert(std::make_shared<BuildOutput>());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
