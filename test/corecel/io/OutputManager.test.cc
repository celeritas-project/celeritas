//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputManager.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/OutputManager.hh"

#include <sstream>

#include "corecel/io/BuildOutput.hh"
#include "corecel/io/ExceptionOutput.hh"
#include "corecel/io/JsonPimpl.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class TestInterface final : public OutputInterface
{
  public:
    TestInterface(Category cat, std::string lab, int value)
        : cat_(cat), label_(lab), value_(value)
    {
    }

    Category    category() const final { return cat_; }
    std::string label() const final { return label_; }

    void output(JsonPimpl* json) const final
    {
#if CELERITAS_USE_JSON
        json->obj = value_;
#else
        (void)sizeof(json);
        (void)sizeof(value_);
#endif
    }

  private:
    Category    cat_{};
    std::string label_{};
    int         value_{};
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OutputManagerTest : public Test
{
  protected:
    using Category = OutputInterface::Category;

    std::string to_string(const OutputManager& om)
    {
        std::ostringstream os;
        om.output(&os);
        return os.str();
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OutputManagerTest, empty)
{
    OutputManager om;

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ("null", result);
    }
    else
    {
        EXPECT_EQ("\"output unavailable\"", result);
    }
}

TEST_F(OutputManagerTest, minimal)
{
    auto first
        = std::make_shared<TestInterface>(Category::input, "input_value", 42);
    auto second = std::make_shared<TestInterface>(Category::result, "out", 1);
    auto third = std::make_shared<TestInterface>(Category::result, "timing", 2);

    OutputManager om;
    om.insert(first);
    om.insert(second);
    om.insert(third);

    EXPECT_THROW(om.insert(first), RuntimeError);

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_EQ(
            R"json({"input":{"input_value":42},"result":{"out":1,"timing":2}})json",
            result);
    }
    else
    {
        EXPECT_EQ("\"output unavailable\"", result);
    }
}

TEST_F(OutputManagerTest, build_output)
{
    OutputManager om;
    om.insert(std::make_shared<celeritas::BuildOutput>());
    std::string result = this->to_string(om);
    EXPECT_TRUE(result.find("CELERITAS_BUILD_TYPE") != std::string::npos)
        << "actual output: " << result;
}

TEST_F(OutputManagerTest, exception_output)
{
    OutputManager om;

    try
    {
        CELER_VALIDATE(false, << "things went wrong");
    }
    catch (...)
    {
        om.insert(std::make_shared<celeritas::ExceptionOutput>(
            std::current_exception()));
    }

    std::string result = this->to_string(om);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_TRUE(result.find("\"what\":\"things went wrong\"")
                    != std::string::npos)
            << "actual output: " << result;
    }
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
