//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ScaleUtils.cc
//---------------------------------------------------------------------------//
#include "ScaleUtils.hh"

#include "orange/orangeinp/CsgTree.hh"

#include "detail/CsgLogicUtils.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
// Build postfix logic with original surface IDs intact
std::vector<logic_int> build_postfix_logic(CsgTree const& tree, NodeId n)
{
    using celeritas::orangeinp::detail::PostfixBuildLogicPolicy;

    PostfixBuildLogicPolicy policy{tree};
    policy(n);
    return std::move(policy).logic();
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
