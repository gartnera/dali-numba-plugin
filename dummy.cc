#include "dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::CPUBackend>::RunImpl(::dali::SampleWorkspace &ws) {
  const auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &output = ws.Output<::dali::CPUBackend>(0);
  /*
  auto shape = input.shape();
  ::dali::TypeInfo type = input.type();
  printf("RunImpl called. shape.num_elements: %ld. shape.sample_dim(): %d. type.name: %s\n",
        shape.num_elements(),
        shape.sample_dim(),
        type.name().c_str()
  );
  */
  auto fn = this->spec_.GetArgument<uint64_t>("fn_ptr");
  ((void (*)(const void*, void*, long))fn)(input.raw_data(), output.raw_mutable_data(), input.size());
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(CustomDummy)
  .DocStr("Make a copy of the input tensor")
  .AddArg("fn_ptr", "function_pointer", ::dali::DALI_INT64 )
  .NumInput(1)
  .NumOutput(1);
