//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ApplyCutoffProcess.cc
//---------------------------------------------------------------------------//
#include "ApplyCutoffProcess.hh"

#include "celeritas/grid/ValueGridBuilder.hh"

#include "ApplyCutoffModel.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
ApplyCutoffProcess::ApplyCutoffProcess(SPConstCutoff cutoffs)
    : cutoffs_(std::move(cutoffs))
{
    CELER_EXPECT(cutoffs_);
}

//---------------------------------------------------------------------------//
auto ApplyCutoffProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<ApplyCutoffModel>(*start_id++, cutoffs_)};
}

//---------------------------------------------------------------------------//
auto ApplyCutoffProcess::step_limits(Applicability range) const
    -> StepLimitBuilders
{
    CELER_EXPECT(range.material);
    CELER_EXPECT(range.particle);

    using VecReal = std::vector<celeritas::real_type>;

    StepLimitBuilders builders;
    builders[celeritas::ValueGridType::macro_xs]
        = std::make_unique<celeritas::ValueGridLogBuilder>(
            range.lower.value(), range.upper.value(), VecReal{1e100, 1e100});
    return builders;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
