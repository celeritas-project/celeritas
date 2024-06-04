//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.cc
//---------------------------------------------------------------------------//
#include "Converter.hh"

#include "corecel/io/Logger.hh"
#include "geocel/detail/LengthUnits.hh"
#include "orange/orangeinp/InputBuilder.hh"

#include "PhysicalVolumeConverter.hh"
#include "ProtoConstructor.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
Converter::Converter(Options&& opts) : opts_{std::move(opts)}
{
    if (!opts_.tol)
    {
        opts_.tol = Tolerance<>::from_default(lengthunits::millimeter);
    }

    if (real_type{1} - ipow<2>(opts_.tol.rel) == real_type{1})
    {
        CELER_LOG(warning)
            << "Requested relative tolerance (" << opts_.tol.rel
            << ") for ORANGE is very small: tracking errors may result due to "
               "incomplete geometry simplification";
    }

    CELER_ENSURE(opts_.tol);
}

//---------------------------------------------------------------------------//
/*!
 * Convert the world.
 */
auto Converter::operator()(arg_type g4world) -> result_type
{
    CELER_EXPECT(g4world);

    using orangeinp::InputBuilder;

    // Convert solids, logical volumes, physical volumes
    PhysicalVolumeConverter::Options options;
    options.verbose = opts_.verbose;
    PhysicalVolumeConverter convert_pv(std::move(options));
    PhysicalVolume world = convert_pv(*g4world);
    CELER_VALIDATE(std::holds_alternative<NoTransformation>(world.transform),
                   << "world volume should not have a transformation");

    // Convert logical volumes into protos
    auto global_proto = ProtoConstructor{opts_.verbose}(*world.lv);

    // Build universes from protos
    result_type result;
    InputBuilder build_input([&opts = opts_] {
        InputBuilder::Options ibo;
        ibo.tol = opts.tol;
        return ibo;
    }());
    result.input = build_input(*global_proto);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
