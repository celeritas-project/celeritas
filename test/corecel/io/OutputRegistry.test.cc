//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputRegistry.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/OutputRegistry.hh"

#include <exception>
#include <regex>
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

    Category category() const final { return cat_; }
    std::string_view label() const final { return label_; }

    void output(JsonPimpl* json) const final { json->obj = value_; }

  private:
    Category cat_{};
    std::string label_{};
    int value_{};
};

//---------------------------------------------------------------------------//
// **IMPORTANT** this class cannot be `final` for exception nesting to work!
// Its members can be, though.
class MockKernelContextException : public RichContextException
{
  public:
    MockKernelContextException(int th, int ev, int tr)
        : thread_(th), event_(ev), track_(tr)
    {
    }

    char const* type() const final { return "MockKernelContextException"; }

    void output(JsonPimpl* json) const final
    {
        json->obj["thread"] = thread_;
        json->obj["event"] = event_;
        json->obj["track"] = track_;
    }

  private:
    int thread_{};
    int event_{};
    int track_{};
};

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class OutputRegistryTest : public Test
{
  protected:
    using Category = OutputInterface::Category;

    std::string to_string(OutputRegistry const& reg)
    {
        static std::regex const file_match(R"re("file":"[^"]+")re");
        static std::regex const line_match(R"re("line":[0-9]+)re");
        std::ostringstream os;
        reg.output(&os);
        std::string result = os.str();
        result = std::regex_replace(result, file_match, R"("file":"FILE")");
        result = std::regex_replace(result, line_match, R"("line":123)");
        return result;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(OutputRegistryTest, empty)
{
    OutputRegistry reg;
    EXPECT_TRUE(reg.empty());

    std::string result = this->to_string(reg);
    EXPECT_EQ("null", result);
}

TEST_F(OutputRegistryTest, minimal)
{
    auto first
        = std::make_shared<TestInterface>(Category::input, "input_value", 42);
    auto second = std::make_shared<TestInterface>(Category::result, "out", 1);
    auto third = std::make_shared<TestInterface>(Category::result, "timing", 2);

    OutputRegistry reg;
    reg.insert(first);
    EXPECT_FALSE(reg.empty());
    reg.insert(second);
    EXPECT_FALSE(reg.empty());
    reg.insert(third);

    EXPECT_THROW(reg.insert(first), RuntimeError);

    EXPECT_JSON_EQ(
        R"json({"input":{"input_value":42},"result":{"out":1,"timing":2}})json",
        this->to_string(reg));
}

TEST_F(OutputRegistryTest, build_output)
{
    OutputRegistry reg;
    reg.insert(std::make_shared<celeritas::BuildOutput>());
    std::string result = this->to_string(reg);
    EXPECT_TRUE(result.find(R"("build_type":)") != std::string::npos)
        << "actual output: " << result;
}

TEST_F(OutputRegistryTest, exception_output)
{
    OutputRegistry reg;
    auto exception_to_output = [&reg](std::exception_ptr const& ep) {
        reg.insert(std::make_shared<celeritas::ExceptionOutput>(ep));
    };

    CELER_TRY_HANDLE(CELER_VALIDATE(false, << "things went wrong"),
                     exception_to_output);

    EXPECT_JSON_EQ(
        R"json({"result":{"exception":{"condition":"false","file":"FILE","line":123,"type":"RuntimeError","what":"things went wrong","which":"runtime"}}})json",
        this->to_string(reg));
}

TEST_F(OutputRegistryTest, nested_exception_output)
{
    OutputRegistry reg;
    auto exception_to_output = [&reg](std::exception_ptr const& ep) {
        reg.insert(std::make_shared<celeritas::ExceptionOutput>(ep));
    };

    CELER_TRY_HANDLE_CONTEXT(CELER_VALIDATE(false, << "things went wrong"),
                             exception_to_output,
                             MockKernelContextException(123, 2, 4567));

    EXPECT_JSON_EQ(
        R"json({"result":{"exception":{"condition":"false","context":{"event":2,"thread":123,"track":4567,"type":"MockKernelContextException"},"file":"FILE","line":123,"type":"RuntimeError","what":"things went wrong","which":"runtime"}}})json",
        this->to_string(reg));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
