//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Device.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/Device.hh"

#include <nlohmann/json.hpp>

#include "corecel/Macros.hh"
#include "corecel/sys/DeviceIO.json.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// NOTE: the device is activated by the googletest 'main' function
class DeviceTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(DeviceTest, json_output)
{
    auto& d = device();
    auto json_out = nlohmann::json(d);
    if (!CELERITAS_USE_CUDA && !CELERITAS_USE_HIP)
    {
        EXPECT_JSON_EQ(json_out.dump(0), "null");
        return;
    }

    ASSERT_TRUE(json_out.count("platform")) << json_out.dump(0);
    auto const& platform = json_out["platform"].get<std::string>();
    if (CELERITAS_USE_CUDA)
    {
        EXPECT_EQ("CUDA", platform);
    }
    else if (CELERITAS_USE_HIP)
    {
        EXPECT_EQ("HIP", platform);
    }

    if (d)
    {
        d.create_streams(10);
        d.destroy_streams();
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
