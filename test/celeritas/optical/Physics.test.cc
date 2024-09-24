//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Physics.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/PhysicsParams.hh"

#include "OpticalMockTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

//---------------------------------------------------------------------------//

class PhysicsParamsTest : public OpticalMockTestBase
{
  protected:
    void SetUp() override {}
};

TEST_F(PhysicsParamsTest, accessors)
{
    PhysicsParams const& p = *this->optical_physics();

    EXPECT_EQ(4, p.num_models());

    std::vector<std::string> model_names;
    std::vector<std::string> model_desc;
    for (auto model_id : range(ModelId{p.num_models()}))
    {
        Model const& m = *p.model(model_id);
        model_names.emplace_back(m.label());
        model_desc.emplace_back(m.description());
    }

    static std::string const expected_model_names[] = {
        "mock-optical-model-1",
        "mock-optical-model-2",
        "mock-optical-model-3",
        "mock-optical-model-4",
    };
    EXPECT_VEC_EQ(expected_model_names, model_names);

    static std::string const expected_model_desc[] = {
        "mock-optical-desc-1",
        "mock-optical-desc-2",
        "mock-optical-desc-3",
        "mock-optical-desc-4",
    };
    EXPECT_VEC_EQ(expected_model_desc, model_desc);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
