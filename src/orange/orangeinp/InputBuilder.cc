//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/InputBuilder.cc
//---------------------------------------------------------------------------//
#include "InputBuilder.hh"

#include <fstream>
#include <nlohmann/json.hpp>

#include "corecel/cont/Range.hh"
#include "corecel/io/JsonPimpl.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "ProtoInterface.hh"

#include "detail/ProtoBuilder.hh"
#include "detail/ProtoMap.hh"

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
void write_protos(detail::ProtoMap const& map, std::string const& filename)
{
    auto result = nlohmann::json(std::vector<std::nullptr_t>(map.size()));
    for (auto univ_id : range(UniverseId{map.size()}))
    {
        JsonPimpl j;
        map.at(univ_id)->output(&j);
        result[univ_id.get()] = std::move(j.obj);
    }

    std::ofstream outf(filename);
    CELER_VALIDATE(outf,
                   << "failed to open output file at \"" << filename << '"');
    outf << result.dump(0);

    CELER_LOG(info) << "Wrote ORANGE protos to " << filename;
}

//---------------------------------------------------------------------------//
//! Helper struct to save JSON to a file
class JsonProtoOutput
{
  public:
    JsonProtoOutput() = default;

    //! Construct with the number of universes
    explicit JsonProtoOutput(UniverseId::size_type size)
    {
        CELER_EXPECT(size > 0);
        output_ = nlohmann::json(std::vector<std::nullptr_t>(size));
    }

    //! Save JSON
    void operator()(UniverseId univ_id, JsonPimpl&& jpo)
    {
        CELER_EXPECT(univ_id < output_.size());
        output_[univ_id.unchecked_get()] = std::move(jpo.obj);
    }

    //! Write debug information to a file
    void write(std::string const& filename) const
    {
        CELER_ASSERT(!output_.empty());
        std::ofstream outf(filename);
        CELER_VALIDATE(
            outf, << "failed to open output file at \"" << filename << '"');
        outf << output_.dump(0);

        CELER_LOG(info) << "Wrote ORANGE debug info to " << filename;
    }

  private:
    nlohmann::json output_;
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
InputBuilder::InputBuilder(Options&& opts) : opts_{std::move(opts)}
{
    CELER_EXPECT(opts_.tol);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE geometry.
 */
auto InputBuilder::operator()(ProtoInterface const& global) const -> result_type
{
    ScopedProfiling profile_this{"build-orange-input"};
    ScopedMem record_mem("orange.build_input");
    CELER_LOG(status) << "Constructing ORANGE surfaces and runtime data";
    ScopedTimeLog scoped_time;

    // Construct the hierarchy of protos
    detail::ProtoMap const protos{global};
    CELER_ASSERT(protos.find(&global) == orange_global_universe);
    if (!opts_.proto_output_file.empty())
    {
        write_protos(protos, opts_.proto_output_file);
    }

    // Build surfaces and metadata
    OrangeInput result;
    JsonProtoOutput debug_outp;
    detail::ProtoBuilder builder(&result, protos, [&] {
        detail::ProtoBuilder::Options pbopts;
        pbopts.tol = opts_.tol;
        if (!opts_.debug_output_file.empty())
        {
            debug_outp = JsonProtoOutput{protos.size()};
            pbopts.save_json = std::ref(debug_outp);
        }
        return pbopts;
    }());
    for (auto univ_id : range(UniverseId{protos.size()}))
    {
        protos.at(univ_id)->build(builder);
    }

    if (!opts_.debug_output_file.empty())
    {
        debug_outp.write(opts_.debug_output_file);
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
