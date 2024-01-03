//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/CsgTestUtils.cc
//---------------------------------------------------------------------------//
#include "CsgTestUtils.hh"

#include <sstream>
#include <variant>

#include "celeritas_config.h"
#include "corecel/io/Join.hh"
#include "corecel/io/Repr.hh"
#include "orange/construct/CsgTree.hh"
#include "orange/orangeinp/detail/CsgUnit.hh"
#include "orange/surf/SurfaceIO.hh"
#include "orange/transform/TransformIO.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "orange/construct/CsgTreeIO.json.hh"
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
std::string to_json_string(CsgTree const& tree)
{
#if CELERITAS_USE_JSON
    nlohmann::json obj{tree};
    return obj.dump();
#else
    CELER_DISCARD(tree);
    return {};
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
