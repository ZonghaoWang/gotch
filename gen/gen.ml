(* Automatically generate the C++ -> C -> Go bindings.
   This takes as input the Descriptions.yaml file that gets generated when
func (Func.c_go_args_list func)  building PyTorch from source.
   Run with: dune exec gen/gen.exe
 *)
open Base
open Stdio

let excluded_functions =
  Set.of_list
    (module String)
    [ "multi_margin_loss"
    ; "multi_margin_loss_out"
    ; "log_softmax_backward_data"
    ; "softmax_backward_data"
    ; "clone"
    ; "copy_"
    ; "conv_transpose2d_backward_out"
    ; "conv_transpose3d_backward_out"
    ; "slow_conv_transpose2d_backward_out"
    ; "slow_conv_transpose3d_backward_out"
    ; "slow_conv3d_backward_out"
    ; "normal"
    ; "_cufft_set_plan_cache_max_size"
    ; "_cufft_clear_plan_cache"
    ; "backward"
    ; "set_data"
    ; "_amp_non_finite_check_and_unscale_"
    ; "_cummin_helper"
    ; "_cummax_helper"
    ; "retain_grad"
    ; "_validate_sparse_coo_tensor_args" ]

let no_tensor_options =
  Set.of_list
    (module String)
    [ "zeros_like"
    ; "empty_like"
    ; "full_like"
    ; "ones_like"
    ; "rand_like"
    ; "randint_like"
    ; "randn_like" ]

(* 
 * let prefixed_functions =
 *   Set.of_list
 *     (module String)
 *     ["add"; "add_"; "div"; "div_"; "mul"; "mul_"; "sub"; "sub_"; "nll_loss"]
 *  *)
let excluded_prefixes = ["_thnn_"; "_th_"; "thnn_"; "th_"; "_foreach"]

let excluded_suffixes = ["_forward"; "_forward_out"]

let yaml_error yaml ~msg =
  Printf.failwithf "%s, %s" msg (Yaml.to_string_exn yaml) ()

let extract_bool = function
  | `Bool b -> b
  | `String "true" -> true
  | `String "false" -> false
  | yaml -> yaml_error yaml ~msg:"expected bool"

let extract_list = function
  | `A l -> l
  | yaml -> yaml_error yaml ~msg:"expected list"

let extract_map = function
  | `O map -> Map.of_alist_exn (module String) map
  | yaml -> yaml_error yaml ~msg:"expected map"

let extract_string = function
  | `String s -> s
  (* The yaml spec for torch uses n which is converted to a bool. *)
  | `Bool b -> if b then "y" else "n"
  | `Float f -> Float.to_string f
  | yaml -> yaml_error yaml ~msg:"expected string"

module Func = struct
  type arg_type =
    | Bool
    | Int64
    | Int64Option
    | Double
    | DoubleOption
    | Tensor
    | TensorOption
    | IntList
    | TensorList
    | TensorOptions
    | Scalar
    | ScalarType
    | Device
    | String

  type arg =
    {arg_name: string; arg_type: arg_type; default_value: string option}

  (* `Func` type   *)
  type t =
    { name: string
    ; args: arg list
    ; returns: [`fixed of int | `dynamic]
    ; (* number of tensors that are returned *)
      kind: [`function_ | `method_] }

  let arg_type_of_string str ~is_nullable =
    match String.lowercase str with
    | "bool" -> Some Bool
    | "int64_t" -> Some (if is_nullable then Int64Option else Int64)
    | "double" -> Some (if is_nullable then DoubleOption else Double)
    | "booltensor" | "indextensor" | "tensor" ->
        Some (if is_nullable then TensorOption else Tensor)
    | "tensoroptions" -> Some TensorOptions
    | "intarrayref" | "intlist" -> Some IntList
    | "tensorlist" -> Some TensorList
    | "device" -> Some Device
    | "scalar" -> Some Scalar
    | "scalartype" -> Some ScalarType
    | "std::string" -> Some String
    | _ -> None

  let c_typed_args_list t =
    List.map t.args ~f:(fun {arg_name; arg_type; _} ->
        match arg_type with
        | IntList ->
            Printf.sprintf "int64_t *%s_data, int %s_len" arg_name arg_name
        | TensorList ->
            Printf.sprintf "tensor *%s_data, int %s_len" arg_name arg_name
        | TensorOptions ->
            Printf.sprintf "int %s_kind, int %s_device" arg_name arg_name
        | String -> Printf.sprintf "char* %s_ptr, int %s_len" arg_name arg_name
        | Int64Option ->
            Printf.sprintf "int64_t %s_v, uint8_t %s_null" arg_name arg_name
        | DoubleOption ->
            Printf.sprintf "double %s_v, uint8_t %s_null" arg_name arg_name
        | otherwise ->
            let simple_type_cstring =
              match otherwise with
              | Bool -> "int"
              | Int64 -> "int64_t"
              | Double -> "double"
              | Tensor -> "tensor"
              | TensorOption -> "tensor"
              | ScalarType -> "int"
              | Device -> "int"
              | Scalar -> "scalar"
              | Int64Option | DoubleOption | String | IntList | TensorList
               |TensorOptions ->
                  assert false
            in
            Printf.sprintf "%s %s" simple_type_cstring arg_name )
    |> String.concat ~sep:", "

  let c_args_list args =
    List.map args ~f:(fun {arg_name; arg_type; _} ->
        match arg_type with
        | Scalar | Tensor -> "*" ^ arg_name
        | TensorOption ->
            Printf.sprintf "(%s ? *%s : torch::Tensor())" arg_name arg_name
        | Bool -> "(bool)" ^ arg_name
        | IntList ->
            Printf.sprintf "torch::IntArrayRef(%s_data, %s_len)" arg_name
              arg_name
        | String ->
            Printf.sprintf "std::string(%s_ptr, %s_len)" arg_name arg_name
        | TensorList ->
            Printf.sprintf "of_carray_tensor(%s_data, %s_len)" arg_name
              arg_name
        | TensorOptions ->
            Printf.sprintf
              "at::device(device_of_int(%s_device)).dtype(at::ScalarType(%s_kind))"
              arg_name arg_name
        | Int64Option ->
            Printf.sprintf
              "%s_null ? c10::nullopt : c10::optional<int64_t>(%s_v)" arg_name
              arg_name
        | DoubleOption ->
            Printf.sprintf
              "%s_null ? c10::nullopt : c10::optional<double>(%s_v)" arg_name
              arg_name
        | ScalarType -> Printf.sprintf "at::ScalarType(%s)" arg_name
        | Device -> Printf.sprintf "device_of_int(%s)" arg_name
        | _ -> arg_name )
    |> String.concat ~sep:", "

  let c_call t =
    match t.kind with
    | `function_ -> Printf.sprintf "torch::%s(%s)" t.name (c_args_list t.args)
    | `method_ -> (
      match t.args with
      | head :: tail ->
          Printf.sprintf "%s->%s(%s)" head.arg_name t.name (c_args_list tail)
      | [] ->
          Printf.failwithf "Method calls should have at least one argument %s"
            t.name () )

  (* 
 *   let replace_map =
 *     Map.of_alist_exn
 *       (module String)
 *       [ ("t", "tr")
 *       ; ("where", "where_")
 *       ; ("view", "view_")
 *       ; ("unsafe", "unsafe_") ]
 *  *)

  let is_method t =
    List.exists t.args ~f:(fun arg ->
        match arg.arg_name with "self" -> true | _ -> false )

  let go_name name =
    let last_underscore name = Str.string_match (Str.regexp ".*_$") name 0 in
    let words = Str.split (Str.regexp "_") name in
    if last_underscore name then
      let cap_words = List.map words ~f:(fun word -> String.capitalize word) in
      String.concat ~sep:"" cap_words ^ "_"
    else
      let cap_words = List.map words ~f:(fun word -> String.capitalize word) in
      String.concat ~sep:"" cap_words

  let go_variable name =
    let goname = go_name name in
    (* NOTE: Deal with Go namespace conflict *)
    let safe_name =
      match goname with
      | "Var" -> "vari"
      | "Unsafe" -> "unsafety"
      | _ -> goname
    in
    String.uncapitalize safe_name

  let c_go_args_list t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        let single_param = Printf.sprintf "%s %s" an in
        match arg.arg_type with
        | Bool -> single_param "int32"
        | Int64 -> single_param "int64"
        | Double -> single_param "float64"
        | Tensor -> single_param "Ctensor"
        | TensorOption -> single_param "Ctensor"
        | Scalar -> single_param "Cscalar"
        | ScalarType -> single_param "int32"
        | Device -> single_param "int32"
        | String -> single_param "string"
        | IntList -> Printf.sprintf "%sData []int64, %sLen int" an an
        | TensorList -> Printf.sprintf "%sData []Ctensor, %sLen int" an an
        | Int64Option -> Printf.sprintf "%sVal int64, %sNull int" an an
        | DoubleOption -> Printf.sprintf "%sVal float64, %sNull int" an an
        | TensorOptions -> Printf.sprintf "%sKind int32, %sDevice int32" an an
    )
    |> String.concat ~sep:", "

  let c_go_args_list_notype t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        let an = match an with "var" -> "vari" | _ -> an in
        let single_param = Printf.sprintf "%s %s" an in
        match arg.arg_type with
        | Bool -> Printf.sprintf "c%s" an
        | Int64 -> Printf.sprintf "c%s" an
        | Double -> Printf.sprintf "c%s" an
        | Tensor -> Printf.sprintf "%s" an
        | TensorOption -> Printf.sprintf "%s" an
        | Scalar -> single_param ""
        | ScalarType -> Printf.sprintf "c%s" an
        | Device -> Printf.sprintf "c%s" an
        | String -> Printf.sprintf "c%s, c%sLen" an an
        | IntList -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | TensorList -> Printf.sprintf "c%sDataPtr, c%sLen" an an
        | Int64Option -> Printf.sprintf "c%sVal, c%sNull" an an
        | DoubleOption -> Printf.sprintf "c%sVal, c%sNull" an an
        | TensorOptions -> Printf.sprintf "c%sKind, c%sDevice" an an )
    |> String.concat ~sep:", "

  (* TODO: convert Go pointer to C pointer *)
  let c_go_args_list_body t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        (* let single_param = Printf.sprintf "%s %s" an in *)
        match arg.arg_type with
        | Bool ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | Int64 ->
            Printf.sprintf "\nc%s := *(*C.int64_t)(unsafe.Pointer(&%s))" an an
        | Double ->
            Printf.sprintf "\nc%s := *(*C.double)(unsafe.Pointer(&%s))" an an
        | Tensor -> ""
        | TensorOption -> ""
        | Scalar -> ""
        | ScalarType ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | Device ->
            Printf.sprintf "\nc%s := *(*C.int)(unsafe.Pointer(&%s))" an an
        | String ->
            Printf.sprintf
              "\n\
               c%s := C.CString(%s)\n\
               %sLen := len(%s)\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an an an
        | IntList ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*C.int64_t)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | TensorList ->
            Printf.sprintf
              "\n\
               c%sDataPtr := (*Ctensor)(unsafe.Pointer(&%sData[0]))\n\
               c%sLen := *(*C.int)(unsafe.Pointer(&%sLen))"
              an an an an
        | Int64Option ->
            Printf.sprintf
              "\n\
               c%sVal := *(*C.int64_t)(unsafe.Pointer(&%sVal))\n\
               c%sNull := *(*C.uint8_t)(unsafe.Pointer(&%sNull))"
              an an an an
        | DoubleOption ->
            Printf.sprintf
              "\n\
               c%sVal := *(*C.double)(unsafe.Pointer(&%sVal))\n\
               c%sNull := *(*C.uint8_t)(unsafe.Pointer(&%sNull))"
              an an an an
        | TensorOptions ->
            Printf.sprintf
              "\n\
               c%sKind := *(*C.int)(unsafe.Pointer(&%sKind))\n\
               c%sDevice := *(*C.int)(unsafe.Pointer(&%sDevice))"
              an an an an )
    |> String.concat ~sep:""

  let self_name = "self"

  let self_tensor arg =
    match arg.arg_type with
    | Tensor -> String.( = ) arg.arg_name self_name
    | _ -> false

  (* 
 *   let type_parameters t =
 *     let needs_scalar_parameter =
 *       List.exists t.args ~f:(fun arg ->
 *           match arg.arg_type with Scalar -> true | _ -> false )
 *     in
 *     let needs_type_parameter =
 *       List.exists t.args ~f:(fun arg ->
 *           match arg.arg_type with
 *           | TensorList | TensorOption -> true
 *           | _ -> false )
 *     in
 *     if needs_type_parameter && needs_scalar_parameter then "Tensor, Scalar"
 *     else if needs_type_parameter then "Tensor"
 *     else if needs_scalar_parameter then "Scalar"
 *     else ""
 *  *)
  
  (* 
 *   let go_args_list t =
 *     (* https://ocaml.janestreet.com/ocaml-core/latest/doc/base/Base/List/#val-partition_tf *)
 *     (* TODO. implement special cases - TensorOptions, ... *)
 *     match List.partition_tf t.args ~f:self_tensor with _, args_list ->
 *       args_list
 *  *)

  let is_inplace t =
    match Str.string_match (Str.regexp ".*_$") t.name 0 with
    | true -> true
    | _ -> false

  let go_typed_args_list t =
    let to_string args =
      let args_list =
        List.map args ~f:(fun arg ->
            let go_arg_type =
              match arg.arg_type with
              | Bool -> "bool"
              | Int64 -> "int64"
              | Double -> "float64"
              | Tensor -> "*Tensor"
              | TensorOption -> "*Tensor"
              | IntList -> "[]int64"
              | TensorList -> "[]Tensor"
              | String -> "string"
              (* TODO. Struct{Kind gotch.DType Device gotch.Device} *)
              (* E.g. `type KindDevice struct{}` *)
              | TensorOptions -> "gotch.KindDevice"
              | Scalar -> "*Scalar"
              | ScalarType -> "gotch.DType"
              | Int64Option -> "[]int64"
              | DoubleOption -> "[]float64"
              | Device -> "gotch.Device"
            in
            match arg.arg_type with
            | TensorOptions ->
                Printf.sprintf "%sKind gotch.DType, %sDevice gotch.Device"
                  (go_variable arg.arg_name) (go_variable arg.arg_name)
            | _ ->
                Printf.sprintf "%s %s" (go_variable arg.arg_name) go_arg_type
        )
      in
      if is_method t && not (is_inplace t) then
        args_list @ ["del bool"] |> String.concat ~sep:", "
      else args_list |> String.concat ~sep:", "
    in
    (* let self_arg = "self Tensor" in *)
    match List.partition_tf t.args ~f:self_tensor with _, args_list ->
      Printf.sprintf "%s" (to_string args_list)

  let go_notype_args_list t =
    let to_string args =
      let args_list =
        List.map args ~f:(fun arg ->
            match arg.arg_type with
            | TensorOptions ->
                Printf.sprintf "%sKind, %sDevice" (go_variable arg.arg_name)
                  (go_variable arg.arg_name)
            | _ -> Printf.sprintf "%s" (go_variable arg.arg_name) )
      in
      if is_method t && not (is_inplace t) then
        args_list @ ["del"] |> String.concat ~sep:", "
      else args_list |> String.concat ~sep:", "
    in
    match List.partition_tf t.args ~f:self_tensor with _, args_list ->
      Printf.sprintf "%s" (to_string args_list)

  let go_return_type t ~fallible =
    (* printf "t name: %s\n" t.name ; *)
    let returns =
      match t.returns with
      | `fixed 1 -> "retVal *Tensor"
      | `fixed v ->
          List.init v ~f:(fun i -> Printf.sprintf "retVal%d *Tensor" i)
          |> String.concat ~sep:", " |> Printf.sprintf "%s"
      | `dynamic -> "retVal []Tensor"
    in
    if is_inplace t then
      if fallible then Printf.sprintf "err error" else Printf.sprintf ""
    else if fallible then Printf.sprintf "%s, err error" returns
    else Printf.sprintf "%s" returns

  let go_return_notype t ~fallible =
    let returns =
      match t.returns with
      | `fixed 1 -> "retVal"
      | `fixed v ->
          List.init v ~f:(fun i -> Printf.sprintf "retVal%d" i)
          |> String.concat ~sep:", " |> Printf.sprintf "%s"
      | `dynamic -> "retVal"
    in
    if is_inplace t then
      if fallible then Printf.sprintf "err" else Printf.sprintf ""
    else if fallible then Printf.sprintf "%s, err" returns
    else Printf.sprintf "%s" returns

  let go_binding_args t =
    List.map t.args ~f:(fun arg ->
        let name = go_variable arg.arg_name in
        match arg.arg_type with
        | Tensor ->
            if String.( = ) name "self" then "ts.ctensor"
            else Printf.sprintf "%s.ctensor" name
        | Scalar -> Printf.sprintf "%s.cscalar" name
        | Bool -> Printf.sprintf "c%s" name
        | ScalarType -> Printf.sprintf "%s.CInt()" name
        | Device -> Printf.sprintf "%s.CInt()" name
        | TensorOptions ->
            Printf.sprintf "%sKind.CInt(), %sDevice.CInt()" name name
        | String -> Printf.sprintf "%s" name
        | IntList -> Printf.sprintf "%s, len(%s)" name name
        | TensorList -> Printf.sprintf "c%s, len(c%s)" name name
        | Int64Option -> Printf.sprintf "c%sVal, c%sNull" name name
        | DoubleOption -> Printf.sprintf "c%sVal, c%sNull" name name
        | TensorOption -> Printf.sprintf "%s.ctensor" name
        | _ -> name )
    |> String.concat ~sep:", "

  let go_binding_body t =
    List.map t.args ~f:(fun arg ->
        let an = go_variable arg.arg_name in
        match arg.arg_type with
        | Bool ->
            Printf.sprintf "c%s := int32(0)\n if %s { c%s = int32(1) }\n" an an
              an
        | Int64 -> ""
        | Double -> ""
        | Tensor -> ""
        | TensorOption -> ""
        | Scalar -> ""
        | ScalarType -> ""
        | Device -> ""
        | String -> ""
        | IntList -> ""
        | Int64Option ->
            Printf.sprintf
              "var c%sVal int64 = 0\n\
              \ var c%sNull int = 1\n\
              \ if len(%s) > 0 {\n\
              \ c%sVal = %s[0]\n\
              \ c%sNull = 0\n\
              \ }\n"
              an an an an an an
        | DoubleOption ->
            Printf.sprintf
              "var c%sVal float64 = 0.0\n\
              \ var c%sNull int = 1\n\
              \ if len(%s) > 0 {\n\
              \ c%sVal = %s[0]\n\
              \ c%sNull = 0\n\
              \ }\n"
              an an an an an an
        | TensorList ->
            Printf.sprintf
              " var c%s []lib.Ctensor\n\
              \  for _, t := range %s {c%s = append(c%s, t.ctensor)}\n"
              an an an an
        | TensorOptions -> "" )
    |> String.concat ~sep:""
end

exception Not_a_simple_arg

let read_yaml filename =
  let funcs =
    (* Split the file to avoid Yaml.of_string_exn segfaulting. *)
    In_channel.with_file filename ~f:In_channel.input_lines
    |> List.group ~break:(fun _ l ->
           String.length l > 0 && Char.( = ) l.[0] '-' )
    |> List.concat_map ~f:(fun lines ->
           Yaml.of_string_exn (String.concat lines ~sep:"\n") |> extract_list
       )
  in
  printf "Read %s, got %d functions.\n%!" filename (List.length funcs) ;
  List.filter_map funcs ~f:(fun yaml ->
      let map = extract_map yaml in
      let name = Map.find_exn map "name" |> extract_string in
      let deprecated = Map.find_exn map "deprecated" |> extract_bool in
      let method_of =
        Map.find_exn map "method_of"
        |> extract_list |> List.map ~f:extract_string
      in
      let arguments = Map.find_exn map "arguments" |> extract_list in
      let returns =
        let is_tensor returns =
          let returns = extract_map returns in
          let return_type =
            Map.find_exn returns "dynamic_type" |> extract_string
          in
          String.( = ) return_type "Tensor"
          || String.( = ) return_type "BoolTensor"
          || String.( = ) return_type "IndexTensor"
        in
        let returns = Map.find_exn map "returns" |> extract_list in
        if List.for_all returns ~f:is_tensor then
          Some (`fixed (List.length returns))
        else
          match returns with
          | [returns] ->
              let return_type =
                Map.find_exn (extract_map returns) "dynamic_type"
                |> extract_string
              in
              if String.( = ) return_type "TensorList" then Some `dynamic
              else None
          | [] | _ :: _ :: _ -> None
      in
      let kind =
        if List.exists method_of ~f:(String.( = ) "namespace") then
          Some `function_
        else if List.exists method_of ~f:(String.( = ) "Tensor") then
          Some `method_
        else None
      in
      if
        (not deprecated)
        && (not
              (List.exists excluded_prefixes ~f:(fun prefix ->
                   String.is_prefix name ~prefix )))
        && (not
              (List.exists excluded_suffixes ~f:(fun suffix ->
                   String.is_suffix name ~suffix )))
        && not (Set.mem excluded_functions name)
      then
        Option.both returns kind
        |> Option.bind ~f:(fun (returns, kind) ->
               try
                 let args =
                   List.filter_map arguments ~f:(fun arg ->
                       let arg = extract_map arg in
                       let arg_name =
                         Map.find_exn arg "name" |> extract_string
                       in
                       let arg_type =
                         Map.find_exn arg "dynamic_type" |> extract_string
                       in
                       let is_nullable =
                         Map.find arg "is_nullable"
                         |> Option.value_map ~default:false ~f:extract_bool
                       in
                       let default_value =
                         Map.find arg "default" |> Option.map ~f:extract_string
                       in
                       match Func.arg_type_of_string arg_type ~is_nullable with
                       | Some Scalar
                         when Option.is_some default_value && not is_nullable
                         ->
                           None
                       | Some TensorOptions
                         when Option.is_some default_value
                              && Set.mem no_tensor_options name ->
                           None
                       | Some arg_type ->
                           let arg_name =
                             match (arg_name, arg_type) with
                             | "self", Scalar -> "self_scalar"
                             | _, _ -> arg_name
                           in
                           Some {Func.arg_name; arg_type; default_value}
                       | None ->
                           if Option.is_some default_value then None
                           else raise Not_a_simple_arg )
                 in
                 Some {Func.name; args; returns; kind}
               with Not_a_simple_arg -> None )
      else None )

let p out_channel s =
  Printf.ksprintf
    (fun line ->
      Out_channel.output_string out_channel line ;
      Out_channel.output_char out_channel '\n' )
    s

let print_inline out_channel s =
  Printf.ksprintf (fun msg -> Out_channel.output_string out_channel msg) s

let write_cpp funcs filename =
  Out_channel.with_file (filename ^ ".cpp.h") ~f:(fun out_cpp ->
      Out_channel.with_file (filename ^ ".h") ~f:(fun out_h ->
          let pc s = p out_cpp s in
          let ph s = p out_h s in
          pc "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
          pc "" ;
          ph "// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
          ph "" ;
          Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
              let c_typed_args_list = Func.c_typed_args_list func in
              match func.returns with
              | `fixed ntensors ->
                  pc "void atg_%s(tensor *out__, %s) {" exported_name
                    c_typed_args_list ;
                  pc "  PROTECT(" ;
                  pc "    auto outputs__ = %s;" (Func.c_call func) ;
                  if ntensors = 1 then
                    pc "    out__[0] = new torch::Tensor(outputs__);"
                  else
                    for i = 0 to ntensors - 1 do
                      pc
                        "    out__[%d] = new \
                         torch::Tensor(std::get<%d>(outputs__));"
                        i i
                    done ;
                  pc "  )" ;
                  pc "}" ;
                  pc "" ;
                  ph "void atg_%s(tensor *, %s);" exported_name
                    c_typed_args_list
              | `dynamic ->
                  pc "tensor *atg_%s(%s) {" exported_name c_typed_args_list ;
                  pc "  PROTECT(" ;
                  pc "    auto outputs__ = %s;" (Func.c_call func) ;
                  (* the returned type is a C++ vector of tensors *)
                  pc "    int sz = outputs__.size();" ;
                  pc
                    "    torch::Tensor **out__ = (torch::Tensor**)malloc((sz \
                     + 1) * sizeof(torch::Tensor*));" ;
                  pc "    for (int i = 0; i < sz; ++i)" ;
                  pc "      out__[i] = new torch::Tensor(outputs__[i]);" ;
                  pc "    out__[sz] = nullptr;" ;
                  pc "    return out__;" ;
                  pc "  )" ;
                  pc "  return nullptr;" ;
                  pc "}" ;
                  pc "" ;
                  ph "tensor *atg_%s(%s);" exported_name c_typed_args_list ) )
  )

let write_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = print_inline out_ml s in
      pm "package tensor" ;
      pm "\n\n" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "\n\n" ;
      pm "// #include \"stdlib.h\"\n" ;
      pm "import \"C\"" ;
      pm "" ;
      pm "\n\n" ;
      pm "import(\n" ;
      pm "  \"unsafe\"\n" ;
      pm "\n" ;
      pm "  \"github.com/zonghaowang/gotch\"\n" ;
      pm "  lib \"github.com/zonghaowang/gotch/libtch\"\n" ;
      pm ")" ;
      pm "\n\n" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
          let is_method = Func.is_method func in
          let is_inplace = Func.is_inplace func in
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let gofunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          let cfunc_name = "lib.Atg" ^ gofunc_name in
          let go_args_list = Func.go_typed_args_list func in
          (* NOTE. temporarily excluding these functions as not implemented at FFI *)
          (* TODO. implement multiple tensors return function []Tensor *)
          let excluded_funcs =
            [ "Chunk"
            ; "AlignTensors"
            ; "BroadcastTensors"
            ; "Meshgrid"
            ; "NonzeroNumpy"
            ; "Split"
            ; "SplitWithSizes"
            ; "Unbind"
            ; "Where"
            ; "Atleast1d1"
            ; "Atleast2d1"
            ; "Atleast3d1"
            ; "Dequantize1"
            ; "QuantizePerTensor1"
            ; "UnsafeChunk"
            ; "UnsafeSplit"
            ; "UnsafeSplitWithSizes"
            ; "AlignTensors" ]
          in
          if
            List.exists excluded_funcs ~f:(fun name ->
                String.( = ) name gofunc_name )
          then pm ""
          else
            match func.returns with
            | `dynamic ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "if del { defer ts.MustDrop() }\n" ;
                pm "  ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "%s(ptr, %s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                (* NOTE. if in_place method, no retVal return *)
                if not (Func.is_inplace func) then
                  pm "  retVal = &Tensor{ctensor: *ptr}\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"
            | `fixed 1 ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func %s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:true) ;
                if is_method && not is_inplace then
                  pm "if del { defer ts.MustDrop() }\n" ;
                pm "  ptr := (*lib.Ctensor)(unsafe.Pointer(C.malloc(0)))\n" ;
                pm "  \n" ;
                pm "  %s" (Func.go_binding_body func) ;
                pm "%s(ptr, %s)\n" cfunc_name (Func.go_binding_args func) ;
                pm "  if err = TorchErr(); err != nil {\n" ;
                pm "    return %s\n"
                  (Func.go_return_notype func ~fallible:true) ;
                pm "  }\n" ;
                (* NOTE. if in_place method, no retVal return *)
                if not (Func.is_inplace func) then
                  pm "  retVal = &Tensor{ctensor: *ptr}\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:true) ;
                pm "} \n"
            | `fixed _ -> pm "" ) ;
      (* TODO. implement for return multiple tensor - []Tensor *)
      pm "// End of implementing Tensor ================================= \n"
  )

let write_must_wrapper funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = print_inline out_ml s in
      pm "package tensor" ;
      pm "\n\n" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "\n\n" ;
      pm "import(\n" ;
      pm "  \"log\"\n" ;
      pm "\n" ;
      pm "  \"github.com/zonghaowang/gotch\"\n" ;
      pm ")" ;
      pm "\n\n" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
          let is_method = Func.is_method func in
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let gofunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          let go_args_list = Func.go_typed_args_list func in
          let go_args_list_notype = Func.go_notype_args_list func in
          (* NOTE. temporarily excluding these functions as not implemented at FFI *)
          (* TODO. implement multiple tensors return function []Tensor *)
          let excluded_funcs =
            [ "Chunk"
            ; "AlignTensors"
            ; "BroadcastTensors"
            ; "Meshgrid"
            ; "NonzeroNumpy"
            ; "Split"
            ; "SplitWithSizes"
            ; "Unbind"
            ; "Where"
            ; "Atleast1d1"
            ; "Atleast2d1"
            ; "Atleast3d1"
            ; "Dequantize1"
            ; "QuantizePerTensor1"
            ; "UnsafeChunk"
            ; "UnsafeSplit"
            ; "UnsafeSplitWithSizes"
            ; "AlignTensors" ]
          in
          if
            List.exists excluded_funcs ~f:(fun name ->
                String.( = ) name gofunc_name )
          then pm ""
          else
            match func.returns with
            | `dynamic ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) %s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `fixed 1 ->
                pm "\n" ;
                if is_method then pm "func(ts *Tensor) Must%s(" gofunc_name
                else pm "func Must%s(" gofunc_name ;
                pm "%s" go_args_list ;
                pm ")(%s) { \n" (Func.go_return_type func ~fallible:false) ;
                pm "  \n" ;
                (* NOTE. No return retVal for in_place method *)
                if Func.is_inplace func then
                  if is_method then
                    pm "  err := ts.%s(%s)\n" gofunc_name go_args_list_notype
                  else pm "  err := %s(%s)\n" gofunc_name go_args_list_notype
                else if is_method then
                  pm "  retVal, err := ts.%s(%s)\n" gofunc_name
                    go_args_list_notype
                else
                  pm "  retVal, err := %s(%s)\n" gofunc_name
                    go_args_list_notype ;
                pm "  if err != nil { log.Fatal(err) }\n" ;
                pm "  \n" ;
                pm "  return %s\n" (Func.go_return_notype func ~fallible:false) ;
                pm "} \n"
            | `fixed _ -> pm "" ) ;
      (* TODO. implement for return multiple tensor - []Tensor *)
      pm "// End of implementing Tensor ================================= \n"
  )

let write_ffi funcs filename =
  Out_channel.with_file filename ~f:(fun out_ml ->
      let pm s = p out_ml s in
      pm "package libtch" ;
      pm "" ;
      pm "// NOTE. THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!" ;
      pm "" ;
      pm "//#include \"stdbool.h\" " ;
      pm "//#include \"torch_api.h\" " ;
      pm "import \"C\"" ;
      pm "" ;
      pm "import \"unsafe\"" ;
      pm "" ;
      Map.iteri funcs ~f:(fun ~key:exported_name ~data:func ->
          (* let is_method = *)
          (* match func.Func.kind with `method_ -> true | `function_ -> false *)
          (* in *)
          (* let is_inplace = *)
          (* Func.is_inplace func *)
          (* 
 *             match exported_name with
 *             | "add_1" -> true
 *             | "sub_1" -> true
 *             | "div_1" -> true
 *             | "mul_1" -> true
 *             | _ -> false
 *  *)
          (* in *)
          (* NOTE. `torch.__PATTERN` *)
          let prefix_2underscore exported_name =
            Str.string_match (Str.regexp "^__") exported_name 0
          in
          (* NOTE. `torch._PATTERN` *)
          let prefix_1underscore exported_name =
            Str.string_match (Str.regexp "^_") exported_name 0
          in
          (* NOTE. `torch.PATTERN_1` *)
          let suffix_1 exported_name =
            Str.string_match (Str.regexp ".*_1$") exported_name 0
          in
          let ffifunc_name =
            if prefix_2underscore exported_name then
              "__" ^ Func.go_name exported_name
            else if prefix_1underscore exported_name then
              "_" ^ Func.go_name exported_name
            else if suffix_1 exported_name then
              Func.go_name exported_name ^ "_"
            else Func.go_name exported_name
          in
          match func.Func.returns with
          | `fixed _ ->
              pm "func Atg%s(ptr *Ctensor, %s){%s \nC.atg_%s(ptr, %s)\n}"
                ffifunc_name (Func.c_go_args_list func)
                (Func.c_go_args_list_body func)
                exported_name
                (Func.c_go_args_list_notype func)
          | `dynamic -> pm ""
          (* TODO: need more implement here *)
          (* pm "func Atg%s(%s)(retValPtr *Ctensor)" *)
          (* (Func.go_name exported_name) *)
          (* (Func.c_go_args_list func)  *) ) )

let methods =
  let c name args = {Func.name; args; returns= `fixed 1; kind= `method_} in
  let ca arg_name arg_type = {Func.arg_name; arg_type; default_value= None} in
  [ c "grad" [ca "self" Tensor]
  ; c "set_requires_grad" [ca "self" Tensor; ca "r" Bool]
  ; c "toType" [ca "self" Tensor; ca "scalar_type" ScalarType]
  ; c "to" [ca "self" Tensor; ca "device" Device] ]

let run ~yaml_filename ~cpp_filename ~ffi_filename ~must_wrapper_filename
    ~wrapper_filename =
  let funcs = read_yaml yaml_filename in
  let funcs = methods @ funcs in
  printf "Generating code for %d functions.\n%!" (List.length funcs) ;
  (* Generate some unique names for overloaded functions. *)
  let funcs =
    List.map funcs ~f:(fun func -> (String.lowercase func.name, func))
    |> Map.of_alist_multi (module String)
    |> Map.to_alist
    |> List.concat_map ~f:(fun (name, funcs) ->
           match funcs with
           | [] -> assert false
           | [func] -> [(name, func)]
           | funcs ->
               List.sort funcs ~compare:(fun (f1 : Func.t) (f2 : Func.t) ->
                   Int.compare (List.length f1.args) (List.length f2.args) )
               |> List.mapi ~f:(fun i func ->
                      ( (if i = 0 then name else Printf.sprintf "%s%d" name i)
                      , func ) ) )
    |> Map.of_alist_exn (module String)
  in
  write_cpp funcs cpp_filename ;
  write_ffi funcs ffi_filename ;
  write_must_wrapper funcs must_wrapper_filename ;
  write_wrapper funcs wrapper_filename

let () =
  run ~yaml_filename:"gen/pytorch/Declarations-v1.7.0.yaml"
    ~cpp_filename:"libtch/torch_api_generated"
    ~ffi_filename:"libtch/c-generated.go"
    ~must_wrapper_filename:"tensor/must-tensor-generated.go"
    ~wrapper_filename:"tensor/tensor-generated.go"
