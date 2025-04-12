//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ScaleUtils.hh
//! \brief API compatibility functions for SCALE (https://scale.ornl.gov/)
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
class CsgTree;

//---------------------------------------------------------------------------//
// Build postfix logic with original surface IDs intact
std::vector<logic_int> build_postfix_logic(CsgTree const& tree, NodeId n);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas