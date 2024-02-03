//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputRegistry.cc
//---------------------------------------------------------------------------//
#include "OutputRegistry.hh"

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>

#include "celeritas_config.h"

#include "OutputInterface.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"

#include "JsonPimpl.hh"
#include "Logger.hh"  // IWYU pragma: keep

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
        = interfaces_[cat].insert({std::move(label), std::move(interface)});
    CELER_VALIDATE(inserted,
                   << "duplicate output entry '" << prev->first
                   << "' for category '" << to_cstring(cat) << "'");
}

//---------------------------------------------------------------------------//
/*!
 * Output all classes to a JSON object.
 */
void OutputRegistry::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
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
            }
            kv.second->output(&json_wrap);
            if (is_global)
            {
                cat_result = std::move(json_wrap.obj);
            }
            else
            {
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
#else
    CELER_DISCARD(j);
    CELER_NOT_CONFIGURED("nljson");
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Output all classes to a JSON object that's written to the given stream.
 */
void OutputRegistry::output(std::ostream* os) const
{
#if CELERITAS_USE_JSON
    JsonPimpl json_wrap;
    this->output(&json_wrap);
    *os << json_wrap.obj.dump();
#else
    // Write a JSON-compatible string and print an explanation to stderr
    CELER_LOG(error) << "Cannot write output to JSON: nljson is not enabled "
                        "in the current build configuration";
    *os << "\"output unavailable\"";
#endif
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
}  // namespace celeritas
